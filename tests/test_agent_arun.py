"""
测试 Agent.arun() 异步方法

验证异步运行的核心功能：基础调用、并发安全、上下文隔离、异常传播。
"""
import asyncio
import time
from unittest.mock import MagicMock, patch
from collections import deque

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from src.violet_agents.core.agent import Agent
from src.violet_agents.core.llm import VioletAgentsLLM
from src.violet_agents.core.config import Config
from src.violet_agents.core.message import Message
from src.violet_agents.core.session import Session


# ---------------------------------------------------------------------------
# 测试辅助：一个实现了 do_run 的具体 Agent
# ---------------------------------------------------------------------------

class MockAgent(Agent):
    """测试用 Agent，调用一次 LLM 后返回结果。"""

    def do_run(self, input_text: str, session: Session, **kwargs) -> Message:
        history = session.get_history()
        user_msg = Message(content=input_text, role="user")
        history.append(user_msg)

        messages = [m.to_openai_dict() for m in history]
        response = self.llm.chat(messages=messages)
        content = response.choices[0].message.content

        assistant_msg = Message(content=content, role="assistant")
        session.add_message(user_msg)
        session.add_message(assistant_msg)
        return assistant_msg


# ---------------------------------------------------------------------------
# Mock 工厂
# ---------------------------------------------------------------------------

def _make_mock_chat_response(content: str) -> ChatCompletion:
    """构造一个假的 ChatCompletion 返回值。"""
    return ChatCompletion(
        id="fake-id",
        created=0,
        model="mock-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                ),
            )
        ],
    )


def _make_agent() -> MockAgent:
    """创建一个带有 mock LLM 的 MockAgent。"""
    llm = MagicMock(spec=VioletAgentsLLM)
    llm.chat.return_value = _make_mock_chat_response("mock response")
    return MockAgent(
        name="TestAgent",
        llm=llm,
        system_prompt="You are a test assistant.",
        config=Config(max_history_length=20),
    )


# ---------------------------------------------------------------------------
# 基础功能测试
# ---------------------------------------------------------------------------

class TestArunBasic:
    """测试 arun() 的基础功能。"""

    @pytest.mark.asyncio
    async def test_arun_returns_message(self):
        """arun() 应返回 Message 对象。"""
        agent = _make_agent()
        result = await agent.arun("hello")
        assert isinstance(result, Message)
        assert result.content == "mock response"
        assert result.role == "assistant"

    @pytest.mark.asyncio
    async def test_arun_accepts_session_id(self):
        """arun() 应接受 session_id 参数。"""
        agent = _make_agent()
        result = await agent.arun("hello", session_id="sess-1")
        assert result.content == "mock response"
        # session 应被创建
        assert "sess-1" in agent.list_sessions()

    @pytest.mark.asyncio
    async def test_arun_multiple_calls_same_session(self):
        """同一 session 多次调用 arun() 应累积历史。"""
        agent = _make_agent()
        await agent.arun("first", session_id="sess-1")
        # 历史中应有 2 条消息（user + assistant）
        history = agent.get_history(session_id="sess-1")
        assert len(history) == 2

        await agent.arun("second", session_id="sess-1")
        history = agent.get_history(session_id="sess-1")
        assert len(history) == 4

    @pytest.mark.asyncio
    async def test_arun_preserves_history_across_sync_and_async(self):
        """同步 run() 和异步 arun() 混合使用时应共享历史。"""
        agent = _make_agent()
        agent.run("sync call", session_id="sess-1")
        await agent.arun("async call", session_id="sess-1")
        history = agent.get_history(session_id="sess-1")
        assert len(history) == 4  # 2 user + 2 assistant


# ---------------------------------------------------------------------------
# 并发与隔离测试
# ---------------------------------------------------------------------------

class TestArunConcurrency:
    """测试 arun() 在并发场景下的正确性。"""

    @pytest.mark.asyncio
    async def test_concurrent_arun_different_sessions(self):
        """不同 session 的并发 arun() 应互不干扰。"""
        agent = _make_agent()

        async def call_with_session(sid: str, text: str) -> Message:
            return await agent.arun(text, session_id=sid)

        # 3 个不同 session 并发调用
        results = await asyncio.gather(
            call_with_session("sess-a", "hello from A"),
            call_with_session("sess-b", "hello from B"),
            call_with_session("sess-c", "hello from C"),
        )

        # 所有结果都应成功返回
        for r in results:
            assert r.content == "mock response"

        # 每个 session 的历史独立
        assert len(agent.get_history(session_id="sess-a")) == 2
        assert len(agent.get_history(session_id="sess-b")) == 2
        assert len(agent.get_history(session_id="sess-c")) == 2

    @pytest.mark.asyncio
    async def test_concurrent_arun_same_session(self):
        """同一 session 的并发 arun() 应共享历史（但需注意竞争条件）。"""
        agent = _make_agent()

        async def call(text: str) -> Message:
            return await agent.arun(text, session_id="shared")

        await asyncio.gather(
            call("msg1"),
            call("msg2"),
            call("msg3"),
        )

        # 同一 session 的历史应包含所有消息（顺序不保证，因为有并发写入）
        history = agent.get_history(session_id="shared")
        assert len(history) >= 2  # 至少有一对 user+assistant

    @pytest.mark.asyncio
    async def test_arun_context_isolation(self):
        """验证每次 arun() 调用使用正确的 contextvars 上下文。

        具体来说：两个并发的 arun() 调用使用不同的 session，
        每个调用内部的 do_run 应该看到正确的 active_session。
        """
        agent = _make_agent()

        # 预先创建 session
        agent.create_session("ctx-test-1")
        agent.create_session("ctx-test-2")

        captured_sessions = {}

        # 包装 do_run 来捕获 active session
        original_do_run = agent.do_run

        def tracking_do_run(input_text: str, session: Session, **kwargs) -> Message:
            active = agent._get_active_session()
            captured_sessions[input_text] = {
                "param_session_id": session.session_id,
                "active_session_id": active.session_id if active else None,
            }
            return original_do_run(input_text, session, **kwargs)

        agent.do_run = tracking_do_run

        async def call_with_session(sid: str) -> Message:
            return await agent.arun(f"msg-to-{sid}", session_id=sid)

        await asyncio.gather(
            call_with_session("ctx-test-1"),
            call_with_session("ctx-test-2"),
        )

        # 验证每个 arun() 调用内部 active_session 与传入的 session 一致
        for key, captured in captured_sessions.items():
            assert captured["param_session_id"] == captured["active_session_id"], (
                f"arun('{key}') 中 active_session ({captured['active_session_id']}) "
                f"与传入的 session ({captured['param_session_id']}) 不一致！"
            )

        # 恢复
        agent.do_run = original_do_run


# ---------------------------------------------------------------------------
# 异常传播测试
# ---------------------------------------------------------------------------

class TestArunException:
    """测试 arun() 的异常传播。"""

    @pytest.mark.asyncio
    async def test_exception_in_do_run_propagates(self):
        """do_run 中抛出的异常应正确传播到 arun() 的调用者。"""
        agent = _make_agent()

        # 让 do_run 抛出异常
        agent.do_run = MagicMock(side_effect=ValueError("test error"))

        with pytest.raises(ValueError, match="test error"):
            await agent.arun("trigger error")

    @pytest.mark.asyncio
    async def test_exception_in_llm_propagates(self):
        """LLM 调用失败应正确传播异常。"""
        agent = _make_agent()
        agent.llm.chat.side_effect = ConnectionError("API unavailable")

        with pytest.raises(ConnectionError, match="API unavailable"):
            await agent.arun("hello")

    @pytest.mark.asyncio
    async def test_arun_without_running_loop(self):
        """在无事件循环的环境下调用 arun()（不 await），应返回协程对象。"""
        agent = _make_agent()
        coro = agent.arun("hello")
        # 不 await 时返回 coroutine 对象
        assert asyncio.iscoroutine(coro)
        await coro  # 清理


# ---------------------------------------------------------------------------
# 性能/耗时测试（确保 arun 不会阻塞事件循环）
# ---------------------------------------------------------------------------

class TestArunNonBlocking:
    """验证 arun() 不会阻塞事件循环。"""

    @pytest.mark.asyncio
    async def test_arun_does_not_block_event_loop(self):
        """并发 arun() 的总耗时应该接近单次耗时，而非累加耗时。"""
        agent = _make_agent()

        # 模拟 LLM 延迟
        agent.llm.chat.side_effect = lambda **kw: (
            time.sleep(0.1),
            _make_mock_chat_response("slow response"),
        )[1]

        start = time.perf_counter()
        await asyncio.gather(
            agent.arun("a", session_id="perf-1"),
            agent.arun("b", session_id="perf-2"),
            agent.arun("c", session_id="perf-3"),
        )
        elapsed = time.perf_counter() - start

        # 并发执行总时间应接近单次耗时（0.1s），而非累加（0.3s）
        # 允许一定浮动
        assert elapsed < 0.25, (
            f"并发 arun 耗时 {elapsed:.2f}s，疑似阻塞了事件循环"
        )
