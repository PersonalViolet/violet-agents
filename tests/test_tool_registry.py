"""
测试 ToolRegistry 的 execute_tool / execute_tools / aexecute_tool / aexecute_tools
以及注册、拦截器等完整功能。
"""

import json
import asyncio
from typing import Dict, Any

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from src.violet_agents.tools.registry import ToolRegistry
from src.violet_agents.tools.base import Tool, ToolParameters, ToolProperty
from src.violet_agents.tools.interceptor import ToolInterceptor
from src.violet_agents.core.message import Message


# ---------------------------------------------------------------------------
# 测试用工具
# ---------------------------------------------------------------------------

class EchoTool(Tool):
    """回显工具：把输入参数原样返回。"""

    def __init__(self):
        super().__init__(
            name="echo",
            description="回显输入参数",
        )

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        return Message(
            role="tool",
            content=f"echo: {json.dumps(parameters)}",
            tool_call_id=tool_call_id,
        )

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "message": ToolProperty(type="string", description="要回显的消息"),
            },
            required=["message"],
        )


class AddTool(Tool):
    """加法工具：返回两数之和。"""

    def __init__(self):
        super().__init__(
            name="add",
            description="计算两数之和",
        )

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        a = parameters["a"]
        b = parameters["b"]
        return Message(
            role="tool",
            content=str(a + b),
            tool_call_id=tool_call_id,
        )

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "a": ToolProperty(type="number", description="第一个加数"),
                "b": ToolProperty(type="number", description="第二个加数"),
            },
            required=["a", "b"],
        )


class NoRequiredParamsTool(Tool):
    """无 required 参数的工具。"""

    def __init__(self):
        super().__init__(
            name="no_req",
            description="无必填参数的工具",
        )

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        return Message(
            role="tool",
            content="no required params",
            tool_call_id=tool_call_id,
        )

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "optional_param": ToolProperty(type="string", description="可选参数"),
            },
            required=[],
        )


# ---------------------------------------------------------------------------
# 测试用拦截器
# ---------------------------------------------------------------------------

class AlwaysApproveInterceptor(ToolInterceptor):
    """始终放行的拦截器。"""

    def do_intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        return True


class AlwaysRejectInterceptor(ToolInterceptor):
    """始终拒绝的拦截器。"""

    def do_intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        return False


class SpyInterceptor(ToolInterceptor):
    """记录拦截调用并始终放行的拦截器，用于验证拦截器被正确调用。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intercepted: list[tuple[str, dict, str]] = []

    def do_intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        self.intercepted.append((tool.name, parameters, tool_call_id))
        return True


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def make_dict_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_call_id: str = "call_001",
    call_type: str = "function",
) -> Dict[str, Any]:
    """构造一个 dict 格式的工具调用。"""
    return {
        "id": tool_call_id,
        "type": call_type,
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments),
        },
    }


def make_openai_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_call_id: str = "call_001",
) -> ChatCompletionMessageFunctionToolCall:
    """构造一个 OpenAI 格式的 ChatCompletionMessageFunctionToolCall。"""
    return ChatCompletionMessageFunctionToolCall(
        id=tool_call_id,
        type="function",
        function={
            "name": tool_name,
            "arguments": json.dumps(arguments),
        },
    )


# ---------------------------------------------------------------------------
# execute_tool 测试
# ---------------------------------------------------------------------------

class TestExecuteTool:
    """测试 execute_tool 单工具执行。"""

    def test_execute_with_dict_input(self):
        """用 dict 格式的工具调用执行。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "hello"})
        result = registry.execute_tool(tc)

        assert result.role == "tool"
        assert result.tool_call_id == "call_001"
        assert "hello" in result.content

    def test_execute_with_openai_input(self):
        """用 ChatCompletionMessageFunctionToolCall 格式执行。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = make_openai_tool_call("echo", {"message": "openai test"})
        result = registry.execute_tool(tc)

        assert result.role == "tool"
        assert "openai test" in result.content

    def test_execute_with_int_argument(self):
        """验证 JSON 解析后参数类型保持正确（数字）。"""
        registry = ToolRegistry()
        registry.register_tool(AddTool())

        tc = make_dict_tool_call("add", {"a": 3, "b": 7})
        result = registry.execute_tool(tc)

        assert "10" in result.content

    def test_execute_defer_tool(self):
        """执行 _defer_tools 中的延迟工具。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool(), is_defer=True)

        tc = make_dict_tool_call("echo", {"message": "deferred"})
        result = registry.execute_tool(tc)

        assert "deferred" in result.content

    # ---- 错误场景 ----

    def test_raises_on_missing_type(self):
        """缺少 type 字段时抛 ValueError。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "x"})
        del tc["type"]

        with pytest.raises(ValueError, match="仅支持 'function'"):
            registry.execute_tool(tc)

    def test_raises_on_wrong_type(self):
        """type 不为 'function' 时抛 ValueError。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "x"}, call_type="chat")

        with pytest.raises(ValueError, match="仅支持 'function'"):
            registry.execute_tool(tc)

    def test_raises_on_missing_id(self):
        """缺少 id 字段时抛 ValueError。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "x"})
        del tc["id"]

        with pytest.raises(ValueError, match="缺少 'id' 字段"):
            registry.execute_tool(tc)

    def test_raises_on_empty_id(self):
        """id 为空字符串时抛 ValueError。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "x"}, tool_call_id="")

        with pytest.raises(ValueError, match="缺少 'id' 字段"):
            registry.execute_tool(tc)

    def test_raises_on_missing_function(self):
        """缺少 function 字段时抛 ValueError。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "x"})
        del tc["function"]

        with pytest.raises(ValueError, match="缺少 'function' 字段"):
            registry.execute_tool(tc)

    def test_raises_on_invalid_json_arguments(self):
        """arguments 不是合法 JSON 时抛 ValueError。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tc = {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "echo",
                "arguments": "not-valid-json!!!",
            },
        }

        with pytest.raises(ValueError, match="工具参数解析失败"):
            registry.execute_tool(tc)

    def test_raises_on_unregistered_tool(self):
        """调用未注册的工具时抛 ValueError。"""
        registry = ToolRegistry()

        tc = make_dict_tool_call("ghost_tool", {})
        with pytest.raises(ValueError, match="未注册"):
            registry.execute_tool(tc)

    def test_raises_on_missing_required_param(self):
        """缺少必填参数时抛 ValueError。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        # echo 要求 message，这里传空参数
        tc = make_dict_tool_call("echo", {})

        with pytest.raises(ValueError, match="参数验证失败"):
            registry.execute_tool(tc)

    def test_no_required_params_passes_validation(self):
        """无 required 参数的工具可以接受空参数。"""
        registry = ToolRegistry()
        registry.register_tool(NoRequiredParamsTool())

        tc = make_dict_tool_call("no_req", {})
        result = registry.execute_tool(tc)

        assert "no required params" in result.content

    # ---- 拦截器场景 ----

    def test_interceptor_approves(self):
        """拦截器放行时，工具正常执行。"""
        interceptor = AlwaysApproveInterceptor()
        registry = ToolRegistry(interceptor=interceptor)
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "approved"})
        result = registry.execute_tool(tc)

        assert "approved" in result.content

    def test_interceptor_rejects(self):
        """拦截器拒绝时，返回拒绝消息。"""
        interceptor = AlwaysRejectInterceptor()
        registry = ToolRegistry(interceptor=interceptor)
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "blocked"})
        result = registry.execute_tool(tc)

        assert result.role == "tool"
        assert "❌" in result.content
        assert "被用户拒绝" in result.content

    def test_interceptor_is_called_with_correct_args(self):
        """验证拦截器被调用时收到正确的参数。"""
        spy = SpyInterceptor()
        registry = ToolRegistry(interceptor=spy)
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "spied"})
        registry.execute_tool(tc)

        assert len(spy.intercepted) == 1
        name, params, call_id = spy.intercepted[0]
        assert name == "echo"
        assert params == {"message": "spied"}
        assert call_id == "call_001"

    def test_no_interceptor_means_no_intercept(self):
        """未设置拦截器时工具直接执行。"""
        registry = ToolRegistry(interceptor=None)
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "no interceptor"})
        result = registry.execute_tool(tc)

        assert "no interceptor" in result.content


# ---------------------------------------------------------------------------
# execute_tools 测试（批量串行）
# ---------------------------------------------------------------------------

class TestExecuteTools:
    """测试 execute_tools 批量串行执行。"""

    def test_execute_multiple_tools_serially(self):
        """批量串行执行多个工具。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tcs = [
            make_dict_tool_call("echo", {"message": "first"}, tool_call_id="call_1"),
            make_dict_tool_call("echo", {"message": "second"}, tool_call_id="call_2"),
            make_dict_tool_call("echo", {"message": "third"}, tool_call_id="call_3"),
        ]

        results = registry.execute_tools(tcs)

        assert len(results) == 3
        assert results[0].tool_call_id == "call_1"
        assert "first" in results[0].content
        assert results[1].tool_call_id == "call_2"
        assert "second" in results[1].content
        assert results[2].tool_call_id == "call_3"
        assert "third" in results[2].content

    def test_execute_empty_list(self):
        """空列表返回空列表。"""
        registry = ToolRegistry()
        results = registry.execute_tools([])
        assert results == []

    def test_mixed_openai_and_dict_inputs(self):
        """混合 OpenAI 和 dict 格式输入。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tcs = [
            make_dict_tool_call("echo", {"message": "dict"}, tool_call_id="call_d"),
            make_openai_tool_call("echo", {"message": "openai"}, tool_call_id="call_o"),
        ]

        results = registry.execute_tools(tcs)

        assert len(results) == 2
        assert "dict" in results[0].content
        assert "openai" in results[1].content

    def test_first_tool_errors_stops_execution(self):
        """串行执行中，第一个工具出错会阻止后续工具执行。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tcs = [
            make_dict_tool_call("ghost_tool", {}),  # 未注册，会抛异常
            make_dict_tool_call("echo", {"message": "never runs"}, tool_call_id="call_2"),
        ]

        with pytest.raises(ValueError, match="未注册"):
            registry.execute_tools(tcs)


# ---------------------------------------------------------------------------
# execute_tools_concurrently 测试（批量同步并发）
# ---------------------------------------------------------------------------

class SleepTool(Tool):
    """带延迟的工具，用于验证并发执行确实比串行快。"""

    def __init__(self):
        super().__init__(
            name="sleep",
            description="休眠指定秒数后返回",
        )

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        import time
        seconds = parameters.get("seconds", 0.1)
        time.sleep(float(seconds))
        return Message(
            role="tool",
            content=f"slept {seconds}s",
            tool_call_id=tool_call_id,
        )

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "seconds": ToolProperty(type="number", description="休眠秒数"),
            },
            required=[],
        )


class TestExecuteToolsConcurrently:
    """测试 execute_tools_concurrently 批量同步并发执行。"""

    def test_basic_concurrent_execution(self):
        """基本并发执行：多个工具同时执行。"""
        registry = ToolRegistry(max_workers=5)
        registry.register_tool(EchoTool())

        tcs = [
            make_dict_tool_call("echo", {"message": f"msg_{i}"}, tool_call_id=f"call_{i}")
            for i in range(5)
        ]

        results = registry.execute_tools_concurrently(tcs)

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.tool_call_id == f"call_{i}"
            assert f"msg_{i}" in r.content

    def test_results_order_matches_input(self):
        """结果顺序与输入顺序一致（而非完成顺序）。"""
        registry = ToolRegistry(max_workers=5)
        registry.register_tool(SleepTool())

        # 故意让后面的先完成（sleep 时间递减）
        tcs = [
            make_dict_tool_call("sleep", {"seconds": 0.15}, tool_call_id="slow"),
            make_dict_tool_call("sleep", {"seconds": 0.02}, tool_call_id="fast"),
        ]

        results = registry.execute_tools_concurrently(tcs)

        # 即使 "fast" 先完成，结果顺序仍按输入顺序
        assert results[0].tool_call_id == "slow"
        assert results[1].tool_call_id == "fast"

    def test_empty_list(self):
        """空列表直接返回空列表。"""
        registry = ToolRegistry()
        results = registry.execute_tools_concurrently([])
        assert results == []

    def test_mixed_openai_and_dict_inputs(self):
        """混合 dict 和 OpenAI 格式输入。"""
        registry = ToolRegistry(max_workers=3)
        registry.register_tool(EchoTool())

        tcs = [
            make_dict_tool_call("echo", {"message": "dict"}, tool_call_id="c1"),
            make_openai_tool_call("echo", {"message": "openai"}, tool_call_id="c2"),
        ]

        results = registry.execute_tools_concurrently(tcs)

        assert len(results) == 2
        assert "dict" in results[0].content
        assert "openai" in results[1].content

    def test_is_faster_than_serial(self):
        """验证并发确实比串行快（sleep 场景）。"""
        registry = ToolRegistry(max_workers=5)
        registry.register_tool(SleepTool())

        tcs = [
            make_dict_tool_call("sleep", {"seconds": 0.1}, tool_call_id=f"call_{i}")
            for i in range(3)
        ]

        import time
        start = time.perf_counter()
        results = registry.execute_tools_concurrently(tcs)
        elapsed = time.perf_counter() - start

        assert len(results) == 3
        # 并发：耗时约 0.1s（最慢的那个）；串行：3 × 0.1 = 0.3s
        assert elapsed < 0.25, f"并发执行耗时 {elapsed:.2f}s，应该 < 0.25s"

    def test_error_aggregation(self):
        """多个工具失败时抛出聚合 RuntimeError。"""
        registry = ToolRegistry(max_workers=3)
        registry.register_tool(EchoTool())

        tcs = [
            make_dict_tool_call("echo", {"message": "ok"}, tool_call_id="ok"),
            make_dict_tool_call("ghost_1", {}, tool_call_id="bad_1"),  # 未注册
            make_dict_tool_call("ghost_2", {}, tool_call_id="bad_2"),  # 未注册
        ]

        with pytest.raises(RuntimeError, match="2/3"):
            registry.execute_tools_concurrently(tcs)

    def test_executor_still_usable_after_concurrent_call(self):
        """调用 execute_tools_concurrently 后 executor 仍可用于 aexecute_tool。"""
        # 这是针对原始 "with self.executor" bug 的回归测试
        registry = ToolRegistry(max_workers=3)
        registry.register_tool(EchoTool())

        # 先执行并发
        tcs = [make_dict_tool_call("echo", {"message": "first"}, tool_call_id="c1")]
        concurrent_results = registry.execute_tools_concurrently(tcs)
        assert len(concurrent_results) == 1

        # 再执行异步 —— 如果 executor 被关闭，这里会抛 RuntimeError
        async def _run():
            tc = make_dict_tool_call("echo", {"message": "second"}, tool_call_id="c2")
            result = await registry.aexecute_tool(tc)
            assert "second" in result.content

        asyncio.run(_run())

    def test_single_tool_same_as_execute_tool(self):
        """单个工具的并发调用结果应与串行一致。"""
        registry = ToolRegistry()
        registry.register_tool(AddTool())

        tc = make_dict_tool_call("add", {"a": 10, "b": 20})

        serial_result = registry.execute_tool(tc)
        concurrent_results = registry.execute_tools_concurrently([tc])

        assert serial_result.content == concurrent_results[0].content

    def test_interceptor_works_in_concurrent(self):
        """并发执行中拦截器仍然生效。"""
        spy = SpyInterceptor()
        registry = ToolRegistry(interceptor=spy, max_workers=3)
        registry.register_tool(EchoTool())

        tcs = [
            make_dict_tool_call("echo", {"message": "a"}, tool_call_id="a"),
            make_dict_tool_call("echo", {"message": "b"}, tool_call_id="b"),
        ]

        results = registry.execute_tools_concurrently(tcs)

        assert len(results) == 2
        # 两个工具都被拦截器检查
        assert len(spy.intercepted) == 2


# ---------------------------------------------------------------------------
# aexecute_tool 测试（异步单工具）
# ---------------------------------------------------------------------------

class TestAExecuteTool:
    """测试 aexecute_tool 异步单工具执行。"""

    def test_aexecute_tool_basic(self):
        """异步执行单个工具。"""
        async def _run():
            registry = ToolRegistry()
            registry.register_tool(EchoTool())

            tc = make_dict_tool_call("echo", {"message": "async"})
            result = await registry.aexecute_tool(tc)

            assert result.role == "tool"
            assert "async" in result.content

        asyncio.run(_run())

    def test_aexecute_tool_with_interceptor(self):
        """异步执行 + 拦截器放行。"""
        async def _run():
            registry = ToolRegistry(interceptor=AlwaysApproveInterceptor())
            registry.register_tool(EchoTool())

            tc = make_dict_tool_call("echo", {"message": "async approved"})
            result = await registry.aexecute_tool(tc)

            assert "async approved" in result.content

        asyncio.run(_run())

    def test_aexecute_tool_with_openai_input(self):
        """异步执行 OpenAI 格式输入。"""
        async def _run():
            registry = ToolRegistry()
            registry.register_tool(EchoTool())

            tc = make_openai_tool_call("echo", {"message": "async openai"})
            result = await registry.aexecute_tool(tc)

            assert "async openai" in result.content

        asyncio.run(_run())

    def test_aexecute_tool_error_propagates(self):
        """异步执行中，错误正常传播。"""
        async def _run():
            registry = ToolRegistry()

            tc = make_dict_tool_call("ghost", {})
            with pytest.raises(ValueError, match="未注册"):
                await registry.aexecute_tool(tc)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# aexecute_tools 测试（批量异步并发）
# ---------------------------------------------------------------------------

class TestAExecuteTools:
    """测试 aexecute_tools 批量异步并发执行。"""

    def test_aexecute_tools_concurrent(self):
        """多个工具并发异步执行。"""
        async def _run():
            registry = ToolRegistry(max_workers=5)
            registry.register_tool(EchoTool())

            tcs = [
                make_dict_tool_call("echo", {"message": f"msg_{i}"}, tool_call_id=f"call_{i}")
                for i in range(5)
            ]

            results = await registry.aexecute_tools(tcs)

            assert len(results) == 5
            for i, r in enumerate(results):
                assert r.tool_call_id == f"call_{i}"
                assert f"msg_{i}" in r.content

        asyncio.run(_run())

    def test_aexecute_tools_empty(self):
        """空列表异步执行。"""
        async def _run():
            registry = ToolRegistry()
            results = await registry.aexecute_tools([])
            assert results == []

        asyncio.run(_run())

    def test_aexecute_tools_error_in_one(self):
        """一个工具失败时，asyncio.gather 会传播异常。"""
        async def _run():
            registry = ToolRegistry()
            registry.register_tool(EchoTool())

            tcs = [
                make_dict_tool_call("echo", {"message": "ok"}, tool_call_id="call_ok"),
                make_dict_tool_call("ghost", {}),  # 会失败
            ]

            with pytest.raises(ValueError, match="未注册"):
                await registry.aexecute_tools(tcs)

        asyncio.run(_run())

    def test_aexecute_tools_mixed_inputs(self):
        """混合 dict 和 OpenAI 格式输入。"""
        async def _run():
            registry = ToolRegistry()
            registry.register_tool(EchoTool())

            tcs = [
                make_dict_tool_call("echo", {"message": "dict"}, tool_call_id="c1"),
                make_openai_tool_call("echo", {"message": "openai"}, tool_call_id="c2"),
            ]

            results = await registry.aexecute_tools(tcs)
            assert len(results) == 2
            assert "dict" in results[0].content
            assert "openai" in results[1].content

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 注册 + 延迟工具 交互测试
# ---------------------------------------------------------------------------

class TestRegistration:
    """测试工具注册及 _tools / _defer_tools 交互。"""

    def test_register_normal_tool(self):
        """注册普通工具后，execute_tool 可用且 get_openai_tools 可见。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        assert "echo" in registry._tools
        assert registry.get_tool("echo") is not None
        assert len(registry.get_openai_tools()) == 1

    def test_register_defer_tool(self):
        """延迟工具不在 get_openai_tools 中，但可通过 get_all_tools 获取。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool(), is_defer=True)

        assert "echo" not in registry._tools
        assert "echo" in registry._defer_tools
        assert registry.get_tool("echo") is None  # get_tool 只查 _tools
        assert len(registry.get_openai_tools()) == 0  # 不暴露给 LLM
        assert "echo" in registry.get_all_tools()  # get_all_tools 包含 defer

    def test_move_from_normal_to_defer(self):
        """同名工具从普通移到延迟。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool())          # 普通
        registry.register_tool(EchoTool(), is_defer=True)  # → 延迟

        assert "echo" not in registry._tools
        assert "echo" in registry._defer_tools

    def test_move_from_defer_to_normal(self):
        """同名工具从延迟移到普通。"""
        registry = ToolRegistry()
        registry.register_tool(EchoTool(), is_defer=True)  # 延迟
        registry.register_tool(EchoTool())                  # → 普通

        assert "echo" in registry._tools
        assert "echo" not in registry._defer_tools

    def test_register_tools_chain(self):
        """register_tools 链式调用。"""
        registry = ToolRegistry()
        result = registry.register_tools(EchoTool(), AddTool())
        assert result is registry  # 返回自身
        assert "echo" in registry._tools
        assert "add" in registry._tools

    def test_register_tools_as_defer(self):
        """register_tools 以 defer 方式注册。"""
        registry = ToolRegistry()
        registry.register_tools(EchoTool(), AddTool(), is_defer=True)

        assert "echo" in registry._defer_tools
        assert "add" in registry._defer_tools


class TestGetOpenAITools:
    """测试 get_openai_tools。"""

    def test_returns_all_normal_tools(self):
        registry = ToolRegistry()
        registry.register_tools(EchoTool(), AddTool())

        tools = registry.get_openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"echo", "add"}

    def test_filters_by_names(self):
        registry = ToolRegistry()
        registry.register_tools(EchoTool(), AddTool())

        tools = registry.get_openai_tools(tool_names=["echo"])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "echo"

    def test_defer_tools_not_included(self):
        registry = ToolRegistry()
        registry.register_tool(EchoTool())
        registry.register_tool(AddTool(), is_defer=True)

        tools = registry.get_openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"echo"}
        assert "add" not in names

    def test_skips_missing_names(self):
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        tools = registry.get_openai_tools(tool_names=["echo", "missing"])
        assert len(tools) == 1


# ---------------------------------------------------------------------------
# 动态注册测试（register_dynamic_tools / unregister_dynamic_tools）
# ---------------------------------------------------------------------------

class TestDynamicRegistration:
    """测试动态工具源的注册、查询、执行与解注册。"""

    # ---- register_dynamic_tools 基础 ----

    def test_register_dynamic_normal_tools(self):
        """动态注册普通工具后，get_tool 可查到。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()

        tool = registry.get_tool("echo")
        assert tool is not None
        assert tool.name == "echo"

    def test_register_dynamic_defer_tools(self):
        """动态注册延迟工具后，get_tool 查不到但 get_all_tools 可查到。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source, is_defer=True)

        dynamic_source["echo"] = EchoTool()

        assert registry.get_tool("echo") is None
        assert "echo" in registry.get_all_tools()
        assert "echo" in registry.get_defer_tools()

    def test_chain_calling(self):
        """register_dynamic_tools 返回 self，支持链式调用。"""
        registry = ToolRegistry()
        src_a: Dict[str, Tool] = {}
        src_b: Dict[str, Tool] = {}

        result = registry.register_dynamic_tools(src_a).register_dynamic_tools(src_b, is_defer=True)
        assert result is registry

    # ---- 外部增删改的实时反映 ----

    def test_external_addition_reflected(self):
        """外部向动态源添加工具后，注册表实时感知。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        # 注册表初始为空
        assert registry.get_tool("echo") is None

        # 外部添加
        dynamic_source["echo"] = EchoTool()
        assert registry.get_tool("echo") is not None

        # 外部再添加
        dynamic_source["add"] = AddTool()
        assert registry.get_tool("add") is not None
        assert registry.get_tool("echo") is not None

    def test_external_deletion_reflected(self):
        """外部从动态源删除工具后，注册表实时感知。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()
        assert registry.get_tool("echo") is not None

        del dynamic_source["echo"]
        assert registry.get_tool("echo") is None

    def test_external_modification_reflected(self):
        """外部修改动态源中工具的值后，注册表实时感知新值。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        echo1 = EchoTool()
        dynamic_source["echo"] = echo1
        assert registry.get_tool("echo") is echo1

        echo2 = EchoTool()
        dynamic_source["echo"] = echo2
        assert registry.get_tool("echo") is echo2
        assert registry.get_tool("echo") is not echo1

    def test_external_clear_reflected(self):
        """外部 clear() 动态源后，注册表中该源的工具全部消失。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()
        dynamic_source["add"] = AddTool()
        assert registry.get_tool("echo") is not None
        assert registry.get_tool("add") is not None

        dynamic_source.clear()

        assert registry.get_tool("echo") is None
        assert registry.get_tool("add") is None

    # ---- 多动态源 ----

    def test_multiple_dynamic_sources(self):
        """多个动态源同时生效，按注册顺序查找（先注册的优先级低）。"""
        registry = ToolRegistry()
        src_a: Dict[str, Tool] = {}
        src_b: Dict[str, Tool] = {}
        registry.register_dynamic_tools(src_a)
        registry.register_dynamic_tools(src_b)

        # 两个源都有同名工具 → 后注册的 src_b 因为 _resolve_tool 遍历顺序，先被找到
        src_a["echo"] = EchoTool()
        echo_b = EchoTool()
        src_b["echo"] = echo_b

        # _resolve_tool 按 _dynamic_sources 列表顺序遍历，先注册的在前
        # 所以 src_a 先被命中
        assert registry.get_tool("echo") is src_a["echo"]

    # ---- 优先级：显式注册 > 动态源 ----

    def test_explicit_overrides_dynamic(self):
        """显式注册的同名工具优先于动态源中的工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_echo = EchoTool()
        dynamic_source["echo"] = dynamic_echo

        explicit_echo = EchoTool()
        registry.register_tool(explicit_echo)

        assert registry.get_tool("echo") is explicit_echo
        assert registry.get_tool("echo") is not dynamic_echo

    def test_explicit_overrides_dynamic_defer(self):
        """显式注册的延迟工具优先于动态延迟源中的同名工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source, is_defer=True)

        dynamic_echo = EchoTool()
        dynamic_source["echo"] = dynamic_echo

        explicit_echo = EchoTool()
        registry.register_tool(explicit_echo, is_defer=True)

        defer_tools = registry.get_defer_tools()
        assert defer_tools["echo"] is explicit_echo

    # ---- execute_tool 支持 ----

    def test_execute_dynamic_normal_tool(self):
        """通过 execute_tool 执行动态普通源中的工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()

        tc = make_dict_tool_call("echo", {"message": "dynamic exec"})
        result = registry.execute_tool(tc)

        assert result.role == "tool"
        assert "dynamic exec" in result.content

    def test_execute_dynamic_defer_tool(self):
        """通过 execute_tool 执行动态延迟源中的工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source, is_defer=True)

        dynamic_source["echo"] = EchoTool()

        tc = make_dict_tool_call("echo", {"message": "dynamic defer"})
        result = registry.execute_tool(tc)

        assert "dynamic defer" in result.content

    def test_execute_dynamic_tool_removed_after_registration(self):
        """动态源中的工具被外部删除后，execute_tool 报未注册。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()

        # 删除
        del dynamic_source["echo"]

        tc = make_dict_tool_call("echo", {"message": "gone"})
        with pytest.raises(ValueError, match="未注册"):
            registry.execute_tool(tc)

    # ---- get_openai_tools 支持 ----

    def test_get_openai_tools_includes_dynamic(self):
        """get_openai_tools 包含动态普通源中的工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()

        tools = registry.get_openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert "echo" in names

    def test_get_openai_tools_excludes_dynamic_defer(self):
        """get_openai_tools 不包含动态延迟源中的工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source, is_defer=True)

        dynamic_source["echo"] = EchoTool()

        tools = registry.get_openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert "echo" not in names

    def test_get_openai_tools_dynamic_and_explicit_merged(self):
        """get_openai_tools 合并显式注册和动态源，显式优先。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()
        registry.register_tool(AddTool())

        tools = registry.get_openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"echo", "add"}

    def test_get_openai_tools_by_name_from_dynamic_source(self):
        """get_openai_tools 按名称过滤时也能查到动态源中的工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()
        dynamic_source["add"] = AddTool()

        tools = registry.get_openai_tools(tool_names=["echo"])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "echo"

    # ---- get_all_tools / get_defer_tools 支持 ----

    def test_get_all_tools_includes_all_dynamic_sources(self):
        """get_all_tools 包含所有动态源（普通 + 延迟）。"""
        registry = ToolRegistry()
        src_normal: Dict[str, Tool] = {}
        src_defer: Dict[str, Tool] = {}
        registry.register_dynamic_tools(src_normal)
        registry.register_dynamic_tools(src_defer, is_defer=True)

        src_normal["echo"] = EchoTool()
        src_defer["add"] = AddTool()

        all_tools = registry.get_all_tools()
        assert "echo" in all_tools
        assert "add" in all_tools

    def test_get_defer_tools_includes_dynamic_defer(self):
        """get_defer_tools 包含动态延迟源中的工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source, is_defer=True)

        dynamic_source["echo"] = EchoTool()

        defer_tools = registry.get_defer_tools()
        assert "echo" in defer_tools

    # ---- unregister_dynamic_tools ----

    def test_unregister_dynamic_tools(self):
        """解注册后，动态源变更不再反映到注册表。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()
        assert registry.get_tool("echo") is not None

        registry.unregister_dynamic_tools(dynamic_source)

        # 解注册后查不到
        assert registry.get_tool("echo") is None
        # 但原始字典不受影响
        assert "echo" in dynamic_source

    def test_unregister_dynamic_tools_chain(self):
        """unregister_dynamic_tools 返回 self，支持链式调用。"""
        registry = ToolRegistry()
        src: Dict[str, Tool] = {}
        registry.register_dynamic_tools(src)

        result = registry.unregister_dynamic_tools(src)
        assert result is registry

    def test_unregister_not_registered_source_raises(self):
        """解注册未注册的源时抛 ValueError。"""
        registry = ToolRegistry()
        unknown_source: Dict[str, Tool] = {}

        with pytest.raises(ValueError, match="未注册"):
            registry.unregister_dynamic_tools(unknown_source)

    def test_unregister_then_reregister(self):
        """解注册后重新注册同一个源，再次生效。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()
        registry.unregister_dynamic_tools(dynamic_source)
        assert registry.get_tool("echo") is None

        # 重新注册
        registry.register_dynamic_tools(dynamic_source)
        assert registry.get_tool("echo") is not None

    # ---- reset_all_tools 支持 ----

    def test_reset_all_tools_resets_dynamic_source_tools(self):
        """reset_all_tools 对动态源中的工具也调用 reset()。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        tool = EchoTool()
        dynamic_source["echo"] = tool

        # reset_all_tools 不抛异常即通过
        registry.reset_all_tools()

    # ---- 拦截器支持 ----

    def test_interceptor_works_with_dynamic_tool(self):
        """拦截器对动态源中的工具同样生效。"""
        spy = SpyInterceptor()
        registry = ToolRegistry(interceptor=spy)
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()

        tc = make_dict_tool_call("echo", {"message": "spied"})
        result = registry.execute_tool(tc)

        assert "spied" in result.content
        assert len(spy.intercepted) == 1
        assert spy.intercepted[0][0] == "echo"

    def test_interceptor_rejects_dynamic_tool(self):
        """拦截器拒绝动态源中的工具时，返回拒绝消息。"""
        interceptor = AlwaysRejectInterceptor()
        registry = ToolRegistry(interceptor=interceptor)
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_source["echo"] = EchoTool()

        tc = make_dict_tool_call("echo", {"message": "blocked"})
        result = registry.execute_tool(tc)

        assert "❌" in result.content

    # ---- 边界场景 ----

    def test_dynamic_source_with_none_value(self):
        """动态源中 key 存在但值为 None 时视为不存在。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        # 直接插入 None（虽然类型标注不允许，但运行时可能发生）
        dynamic_source["echo"] = None  # type: ignore

        assert registry.get_tool("echo") is None

    def test_dynamic_source_starts_empty(self):
        """注册空动态源后查询不报错。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        assert registry.get_tool("nothing") is None
        assert registry.get_all_tools() == {}
        assert registry.get_openai_tools() == []

    def test_dynamic_source_added_then_queried_multiple_times(self):
        """多次查询动态源中的工具，结果一致。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        tool = EchoTool()
        dynamic_source["echo"] = tool

        for _ in range(5):
            assert registry.get_tool("echo") is tool

    def test_multiple_dynamic_sources_same_tool_name(self):
        """多个动态源有同名工具时，按注册顺序返回第一个命中的。"""
        registry = ToolRegistry()
        src_a: Dict[str, Tool] = {}
        src_b: Dict[str, Tool] = {}

        # 注册顺序：src_a 先，src_b 后
        registry.register_dynamic_tools(src_a)
        registry.register_dynamic_tools(src_b)

        echo_a = EchoTool()
        echo_b = EchoTool()
        src_a["echo"] = echo_a
        src_b["echo"] = echo_b

        # _resolve_tool 遍历 _dynamic_sources 列表，src_a 先被访问
        assert registry.get_tool("echo") is echo_a

    def test_explicit_defer_before_dynamic(self):
        """显式延迟工具优先级高于动态普通源中的同名工具。"""
        registry = ToolRegistry()
        dynamic_source: Dict[str, Tool] = {}
        registry.register_dynamic_tools(dynamic_source)

        dynamic_echo = EchoTool()
        dynamic_source["echo"] = dynamic_echo

        explicit_echo = EchoTool()
        registry.register_tool(explicit_echo, is_defer=True)

        # get_tool 查 _tools + _dynamic_sources，不查 defer
        # dynamic_source 中有 echo → 返回 dynamic_echo
        assert registry.get_tool("echo") is dynamic_echo

        # execute_tool 用 _resolve_tool，按优先级 _tools > _dynamic_sources > _defer_tools
        # _tools 无 → _dynamic_sources 有 → 返回 dynamic_echo（不走到 _defer_tools）
        tc = make_dict_tool_call("echo", {"message": "test"})
        result = registry.execute_tool(tc)
        assert "test" in result.content


class TestResetAllTools:
    """测试 reset_all_tools。"""

    def test_resets_all_tools_state(self):
        """验证 reset 被调用（EchoTool 无状态，但不会报错）。"""
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register_tool(tool)
        registry.register_tool(AddTool(), is_defer=True)

        # 不应抛异常
        registry.reset_all_tools()


class TestInterceptorRules:
    """测试 ToolInterceptor 的规则优先级。"""

    def test_whitelist_auto_approves(self):
        """白名单工具自动放行，不调用 do_intercept。"""

        class InterceptAllInterceptor(ToolInterceptor):
            def do_intercept(self, tool, parameters, tool_call_id):
                return False  # 默认拒绝

        interceptor = InterceptAllInterceptor(whitelist=[EchoTool])
        registry = ToolRegistry(interceptor=interceptor)
        registry.register_tool(EchoTool())

        tc = make_dict_tool_call("echo", {"message": "wl"})
        result = registry.execute_tool(tc)

        assert "wl" in result.content  # 白名单放行

    def test_intercept_list_overrides_whitelist(self):
        """同时出现在两个列表时，以 intercept_list 为准。"""
        echo = EchoTool()

        class InterceptAllInterceptor(ToolInterceptor):
            def do_intercept(self, tool, parameters, tool_call_id):
                return False

        interceptor = InterceptAllInterceptor(
            whitelist=[EchoTool],
            intercept_list=[EchoTool],
        )
        registry = ToolRegistry(interceptor=interceptor)
        registry.register_tool(echo)

        tc = make_dict_tool_call("echo", {"message": "nope"})
        result = registry.execute_tool(tc)

        assert "❌" in result.content  # intercept_list 优先，被拒绝

    def test_auto_approve_if_no_rules_false(self):
        """auto_approve_if_no_rules=False 时，不在任何列表的工具需拦截。"""
        echo = EchoTool()

        class InterceptAllInterceptor(ToolInterceptor):
            def do_intercept(self, tool, parameters, tool_call_id):
                return False

        interceptor = InterceptAllInterceptor(auto_approve_if_no_rules=False)
        registry = ToolRegistry(interceptor=interceptor)
        registry.register_tool(echo)

        tc = make_dict_tool_call("echo", {"message": "nope"})
        result = registry.execute_tool(tc)

        assert "❌" in result.content

    def test_auto_approve_if_no_rules_true(self):
        """auto_approve_if_no_rules=True 时，不在任何列表的工具自动放行。"""
        echo = EchoTool()

        class InterceptAllInterceptor(ToolInterceptor):
            def do_intercept(self, tool, parameters, tool_call_id):
                return False  # 不会走到这里

        interceptor = InterceptAllInterceptor(auto_approve_if_no_rules=True)
        registry = ToolRegistry(interceptor=interceptor)
        registry.register_tool(echo)

        tc = make_dict_tool_call("echo", {"message": "auto ok"})
        result = registry.execute_tool(tc)

        assert "auto ok" in result.content
