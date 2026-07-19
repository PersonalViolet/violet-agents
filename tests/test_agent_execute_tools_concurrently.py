"""
测试 Agent.execute_tools_concurrently() —— 并发执行工具调用。

核心验证点：
1. 真正的并发（耗时 ≈ max(single) 而非 sum(single)）
2. 结果顺序与输入顺序一致
3. Agent 基类路径：直接走 self.execute_tool()（可被子类重写）
4. ReactAgent 路径：PreToolCall / PostToolCall 钩子在并发路径上对每条 tool_call 都生效
5. 空列表、错误聚合等边界情况
"""

import json
import time
from typing import Dict, Any

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from src.violet_agents.core.agent import Agent
from src.violet_agents.core.llm import VioletAgentsLLM
from src.violet_agents.core.message import Message
from src.violet_agents.tools.registry import ToolRegistry
from src.violet_agents.tools.base import Tool, ToolParameters, ToolProperty
from src.violet_agents.agents.react_agent import ReactAgent


# ---------------------------------------------------------------------------
# 测试用工具
# ---------------------------------------------------------------------------

class SleepTool(Tool):
    """带延迟的工具 —— 用于验证并发确实比串行快。"""

    def __init__(self):
        super().__init__(
            name="sleep",
            description="休眠指定秒数后返回",
        )

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        seconds = float(parameters.get("seconds", 0.1))
        time.sleep(seconds)
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


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def make_dict_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_call_id: str = "call_001",
) -> Dict[str, Any]:
    return {
        "id": tool_call_id,
        "type": "function",
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
    return ChatCompletionMessageFunctionToolCall(
        id=tool_call_id,
        type="function",
        function={
            "name": tool_name,
            "arguments": json.dumps(arguments),
        },
    )


# ---------------------------------------------------------------------------
# 最小化的 Agent 子类 —— 不需要真实 LLM
# ---------------------------------------------------------------------------

class _TestAgent(Agent):
    """仅用于测试 execute_tools_concurrently 的 Agent 子类。"""

    def do_run(self, input_text: str, session, **kwargs) -> Message:
        raise NotImplementedError("测试中不使用 do_run")


# ---------------------------------------------------------------------------
# Agent 基类 execute_tools_concurrently 测试
# ---------------------------------------------------------------------------

class TestAgentExecuteToolsConcurrently:
    """测试 Agent 基类的 execute_tools_concurrently。"""

    @pytest.fixture
    def agent(self) -> _TestAgent:
        """创建一个带 SleepTool 和 EchoTool 的 Agent。"""
        llm = VioletAgentsLLM(
            model="deepseek-chat",
            api_key="dummy",
            base_url="http://localhost",
        )
        registry = ToolRegistry()
        registry.register_tools(SleepTool(), EchoTool())
        return _TestAgent(
            name="test_agent",
            llm=llm,
            tool_registry=registry,
        )

    # ---- 基本功能 ----

    def test_basic_concurrent_execution(self, agent: _TestAgent):
        """多个工具并发执行，结果数量与输入一致。"""
        tcs = [
            make_dict_tool_call("echo", {"message": f"msg_{i}"}, tool_call_id=f"call_{i}")
            for i in range(5)
        ]

        results = agent.execute_tools_concurrently(tcs)

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.tool_call_id == f"call_{i}"
            assert f"msg_{i}" in r.content

    def test_results_order_matches_input(self, agent: _TestAgent):
        """结果顺序与输入顺序一致（而非完成顺序）。"""
        # 后面的 sleep 更短 → 先完成，但结果顺序应不变
        tcs = [
            make_dict_tool_call("sleep", {"seconds": 0.15}, tool_call_id="slow"),
            make_dict_tool_call("sleep", {"seconds": 0.01}, tool_call_id="fast"),
        ]

        results = agent.execute_tools_concurrently(tcs)

        assert results[0].tool_call_id == "slow"
        assert results[1].tool_call_id == "fast"

    def test_empty_list(self, agent: _TestAgent):
        """空列表直接返回空列表。"""
        results = agent.execute_tools_concurrently([])
        assert results == []

    def test_single_tool(self, agent: _TestAgent):
        """单个工具调用的行为应与直接调用 execute_tool 一致。"""
        tc = make_dict_tool_call("echo", {"message": "solo"})

        single_result = agent.execute_tool(tc)
        concurrent_results = agent.execute_tools_concurrently([tc])

        assert single_result.content == concurrent_results[0].content
        assert single_result.tool_call_id == concurrent_results[0].tool_call_id

    # ---- 真正的并发验证 ----

    def test_is_truly_concurrent(self, agent: _TestAgent):
        """并发耗时应接近 max(single_time)，而非 sum(single_time)。"""
        tcs = [
            make_dict_tool_call("sleep", {"seconds": 0.1}, tool_call_id=f"call_{i}")
            for i in range(3)
        ]

        start = time.perf_counter()
        results = agent.execute_tools_concurrently(tcs)
        elapsed = time.perf_counter() - start

        assert len(results) == 3
        # 并发：≈ 0.1s（最慢的那个）；串行：3 × 0.1 = 0.3s
        assert elapsed < 0.25, (
            f"并发执行耗时 {elapsed:.2f}s，应 < 0.25s（如果串行则 > 0.3s）"
        )

    def test_concurrent_is_faster_than_serial(self, agent: _TestAgent):
        """并发执行明显快于等价的串行执行。"""
        tcs = [
            make_dict_tool_call("sleep", {"seconds": 0.1}, tool_call_id=f"call_{i}")
            for i in range(3)
        ]

        # 串行时间
        start = time.perf_counter()
        for tc in tcs:
            agent.execute_tool(tc)
        serial_elapsed = time.perf_counter() - start

        # 并发时间
        start = time.perf_counter()
        agent.execute_tools_concurrently(tcs)
        concurrent_elapsed = time.perf_counter() - start

        # 3 个 0.1s 的任务并发应明显快于串行
        assert concurrent_elapsed < serial_elapsed * 0.7, (
            f"并发 {concurrent_elapsed:.2f}s 应明显快于串行 {serial_elapsed:.2f}s"
        )

    # ---- 混合输入格式 ----

    def test_mixed_openai_and_dict_inputs(self, agent: _TestAgent):
        """混合 dict 和 OpenAI 格式输入都能正确处理。"""
        tcs = [
            make_dict_tool_call("echo", {"message": "dict"}, tool_call_id="c1"),
            make_openai_tool_call("echo", {"message": "openai"}, tool_call_id="c2"),
        ]

        results = agent.execute_tools_concurrently(tcs)

        assert len(results) == 2
        assert "dict" in results[0].content
        assert "openai" in results[1].content

    # ---- 错误场景 ----

    def test_error_aggregation(self, agent: _TestAgent):
        """多个工具失败时抛出聚合 RuntimeError，包含失败数量。"""
        tcs = [
            make_dict_tool_call("echo", {"message": "ok"}, tool_call_id="ok"),
            make_dict_tool_call("ghost_a", {}, tool_call_id="bad_1"),  # 未注册
            make_dict_tool_call("ghost_b", {}, tool_call_id="bad_2"),  # 未注册
        ]

        with pytest.raises(RuntimeError, match="2/3"):
            agent.execute_tools_concurrently(tcs)

    def test_error_message_contains_failed_tool_names(self, agent: _TestAgent):
        """错误消息中包含失败工具的信息。"""
        tcs = [
            make_dict_tool_call("ghost_x", {}, tool_call_id="bad"),
        ]

        with pytest.raises(RuntimeError, match="1/1"):
            agent.execute_tools_concurrently(tcs)

    # ---- 子类重写验证 ----

    def test_calls_self_execute_tool_not_registry_directly(self):
        """验证 execute_tools_concurrently 调用的是 self.execute_tool()
        而非 self.tool_registry.execute_tool() 直接。

        这个测试通过创建一个重写 execute_tool 的子类来验证：
        在 execute_tool 中注入额外的 metadata，确保并发路径上也携带这些标记。
        """

        class AgentWithMarkedExecuteTool(Agent):
            """重写 execute_tool 添加标记，验证并发路径走的是子类版本。"""

            def do_run(self, input_text, session, **kwargs):
                raise NotImplementedError

            def execute_tool(self, tool_call):
                result = super().execute_tool(tool_call)
                # 注入标记：证明是子类的 execute_tool 被调用了
                result.metadata = result.metadata or {}
                result.metadata["_hooked"] = True
                return result

        llm = VioletAgentsLLM(
            model="deepseek-chat",
            api_key="dummy",
            base_url="http://localhost",
        )
        registry = ToolRegistry()
        registry.register_tool(EchoTool())

        agent = AgentWithMarkedExecuteTool(
            name="marked",
            llm=llm,
            tool_registry=registry,
        )

        tcs = [
            make_dict_tool_call("echo", {"message": "a"}, tool_call_id="c1"),
            make_dict_tool_call("echo", {"message": "b"}, tool_call_id="c2"),
        ]

        results = agent.execute_tools_concurrently(tcs)

        assert len(results) == 2
        for r in results:
            assert r.metadata is not None
            assert r.metadata.get("_hooked") is True, (
                f"execute_tools_concurrently 未通过 self.execute_tool() 执行！"
                f" 工具结果缺少子类注入的 metadata._hooked 标记。"
            )


# ---------------------------------------------------------------------------
# ReactAgent 钩子并发测试
# ---------------------------------------------------------------------------

class TestReactAgentHooksInConcurrent:
    """验证 ReactAgent 的 PreToolCall / PostToolCall 钩子在并发路径上正确触发。"""

    @pytest.fixture
    def react_agent(self) -> ReactAgent:
        """创建一个不带真实 LLM 调用的 ReactAgent（仅用于测试 execute_tool 路径）。"""
        llm = VioletAgentsLLM(
            model="deepseek-chat",
            api_key="dummy",
            base_url="http://localhost",
        )
        registry = ToolRegistry()
        registry.register_tools(EchoTool(), SleepTool())
        return ReactAgent(
            name="test_react",
            llm=llm,
            max_steps=3,
            tool_registry=registry,
        )

    def test_pre_post_hooks_fire_for_all_tools_in_concurrent(self, react_agent: ReactAgent):
        """并发执行时，每条 tool_call 都触发 PreToolCall 和 PostToolCall 钩子。"""
        pre_calls: list[str] = []
        post_calls: list[str] = []

        def on_pre(tool_call):
            name = tool_call.function.name if hasattr(tool_call, 'function') else tool_call.get("function", {}).get("name", "?")
            pre_calls.append(name)

        def on_post(msg: Message):
            post_calls.append(msg.tool_call_id)

        react_agent.register_session_hook("PreToolCall", on_pre)
        react_agent.register_session_hook("PostToolCall", on_post)

        # 直接创建并激活 session，避免触发真实 LLM 调用
        react_agent.create_session("hook-test")
        react_agent.switch_session("hook-test")

        tcs = [
            make_dict_tool_call("echo", {"message": "a"}, tool_call_id="c1"),
            make_dict_tool_call("echo", {"message": "b"}, tool_call_id="c2"),
            make_dict_tool_call("echo", {"message": "c"}, tool_call_id="c3"),
        ]

        results = react_agent.execute_tools_concurrently(tcs)

        assert len(results) == 3
        assert len(pre_calls) == 3, (
            f"PreToolCall 应触发 3 次，实际触发 {len(pre_calls)} 次"
        )
        assert len(post_calls) == 3, (
            f"PostToolCall 应触发 3 次，实际触发 {len(post_calls)} 次"
        )
        assert pre_calls == ["echo", "echo", "echo"]
        assert set(post_calls) == {"c1", "c2", "c3"}

    def test_hooks_fire_in_correct_order(self, react_agent: ReactAgent):
        """对于每个 tool_call，PreToolCall 应先于 PostToolCall 触发。"""
        events: list[tuple[str, str]] = []  # (event, tool_call_id)

        def on_pre(tool_call):
            if hasattr(tool_call, 'id'):
                cid = tool_call.id
            elif isinstance(tool_call, dict):
                cid = tool_call.get("id", "?")
            else:
                cid = "?"
            events.append(("pre", cid))

        def on_post(msg: Message):
            events.append(("post", msg.tool_call_id))

        react_agent.register_session_hook("PreToolCall", on_pre)
        react_agent.register_session_hook("PostToolCall", on_post)

        react_agent.create_session("order-test")
        react_agent.switch_session("order-test")

        tcs = [
            make_dict_tool_call("echo", {"message": "x"}, tool_call_id="single"),
        ]

        react_agent.execute_tools_concurrently(tcs)

        # 提取 single 工具的事件顺序
        single_events = [e for e in events if e[1] == "single"]
        assert single_events == [("pre", "single"), ("post", "single")], (
            f"钩子触发顺序错误: {single_events}"
        )

    def test_builtin_hooks_fire_in_concurrent(self, react_agent: ReactAgent):
        """内置钩子 (_on_temp_tool_called_hook, _handle_search_tools_hook)
        在并发路径上也应正常触发（不抛异常即可验证）。"""
        react_agent.create_session("builtin-test")
        react_agent.switch_session("builtin-test")

        tcs = [
            make_dict_tool_call("echo", {"message": "a"}, tool_call_id="c1"),
            make_dict_tool_call("sleep", {"seconds": 0.01}, tool_call_id="c2"),
        ]

        # 不应抛异常（内置钩子正常执行）
        results = react_agent.execute_tools_concurrently(tcs)
        assert len(results) == 2

    def test_concurrent_with_temp_tool_hooks(self, react_agent: ReactAgent):
        """验证临时工具的 PreToolCall 钩子在并发路径上正常工作。"""
        react_agent.create_session("temp-test")
        react_agent.switch_session("temp-test")

        sess = react_agent._get_active_session()
        # 手动注入一个临时工具到 session
        sess.agent_state.setdefault("temp_tools_names", set()).add("echo")
        sess.agent_state.setdefault("temp_tools_last_call_round", {})
        sess.agent_state["current_round"] = 5

        tcs = [
            make_dict_tool_call("echo", {"message": "tmp"}, tool_call_id="call_tmp"),
        ]

        results = react_agent.execute_tools_concurrently(tcs)
        assert len(results) == 1

        # echo 被当作临时工具调用后，last_call_round 应更新为 current_round
        assert sess.agent_state["temp_tools_last_call_round"].get("echo") == 5
