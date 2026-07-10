"""Agent 线程安全测试

测试 Agent 基类的线程安全机制：
1. ContextVar 隔离：每个线程拥有独立的 active session、消息历史、默认 session ID
2. RLock 保护：Session 池 (_sessions) 的并发读写安全
3. run() 并发：多线程同时调用 agent.run() 互不干扰
4. 工具状态隔离：Tool 的 ContextVar 状态天然线程隔离
"""

import pytest
import threading
import time
import contextvars
from unittest.mock import MagicMock, patch, PropertyMock
from collections import deque

from src.violet_agents.core.agent import Agent
from src.violet_agents.core.llm import VioletAgentsLLM
from src.violet_agents.core.message import Message
from src.violet_agents.core.config import Config
from src.violet_agents.tools.registry import ToolRegistry
from src.violet_agents.tools.base import Tool, ToolParameters, ToolProperty


# ---------------------------------------------------------------------------
# Mock 辅助
# ---------------------------------------------------------------------------

class FakeChoiceMessage:
    """模拟 OpenAI ChatCompletionMessage"""
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role
        self.tool_calls = None


class FakeChoice:
    """模拟 OpenAI Choice"""
    def __init__(self, content: str):
        self.message = FakeChoiceMessage(content)


class FakeChatCompletion:
    """模拟 OpenAI ChatCompletion"""
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


def make_mock_llm(response_content: str = "mock response"):
    """创建一个 mock 的 VioletAgentsLLM，其 chat() 返回指定的响应。"""
    mock = MagicMock(spec=VioletAgentsLLM)
    mock.chat.return_value = FakeChatCompletion(response_content)
    mock.model = "mock-model"
    # 以下属性在 Agent.__init__ 中不会被访问（Agent 不检查 llm 内部），安全
    return mock


def _make_mock_llm_class():
    """通过替换 __init__ 创建一个完全可实例化的 mock LLM 类。

    避免调用真实 VioletAgentsLLM.__init__（它会尝试读取环境变量并创建 OpenAI 客户端）。
    """
    class MockLLM(VioletAgentsLLM):
        def __init__(self, response_content: str = "mock response", **kwargs):
            # 跳过父类 __init__，直接设置所需属性
            self.model = kwargs.get("model", "mock-model")
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens", None)
            self.provider = kwargs.get("provider", "deepseek")
            self._response_content = response_content

        def chat(self, messages=None, tools=None, tool_choice='none', **kwargs):
            return FakeChatCompletion(self._response_content)

    return MockLLM


# ---------------------------------------------------------------------------
# 一个可测试的 Agent 具体实现（与 SimpleAgent 类似，但不依赖真实 LLM）
# ---------------------------------------------------------------------------

class _TestAgent(Agent):
    """测试用的 Agent 子类，do_run 简单返回 mock 响应。"""

    def __init__(self, name="test", llm=None, system_prompt=None, config=None,
                 tool_registry=None, mock_response="hello"):
        if llm is None:
            MockLLM = _make_mock_llm_class()
            llm = MockLLM(response_content=mock_response)
        super().__init__(name, llm, system_prompt, config, tool_registry)

    def do_run(self, input_text: str, session, **kwargs) -> Message:
        sess = session
        user_msg = Message(content=input_text, role="user")
        sess.add_message(user_msg)

        response = self.llm.chat(messages=list(sess.get_history()))
        response_text = response.choices[0].message.content
        resp_msg = Message(content=response_text, role="assistant")
        sess.add_message(resp_msg)
        return resp_msg


# ---------------------------------------------------------------------------
# 测试：ContextVar 隔离
# ---------------------------------------------------------------------------

class TestContextVarIsolation:
    """验证 ContextVar 在每线程中的隔离性。"""

    def test_active_session_is_none_by_default(self):
        """新创建的 Agent，当前上下文没有 active session。"""
        agent = _TestAgent()
        assert agent._get_active_session() is None

    def test_active_session_isolated_per_thread(self):
        """不同线程的 active session 彼此独立。"""
        agent = _TestAgent()
        results = {}

        def thread_fn(thread_id: str):
            sid = agent.create_session(f"session-{thread_id}")
            agent.switch_session(sid)
            # 在当前线程中验证
            active = agent._get_active_session()
            results[thread_id] = active.session_id if active else None

        t1 = threading.Thread(target=thread_fn, args=("A",))
        t2 = threading.Thread(target=thread_fn, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["A"] == "session-A"
        assert results["B"] == "session-B"

        # 主线程应该仍然没有 active session
        assert agent._get_active_session() is None

    def test_default_session_id_isolated_per_thread(self):
        """不同线程的默认 session ID 彼此独立。"""
        agent = _TestAgent()
        results = {}

        def thread_fn(thread_id: str):
            agent._set_default_session_id(f"default-{thread_id}")
            results[thread_id] = agent._get_default_session_id()

        t1 = threading.Thread(target=thread_fn, args=("A",))
        t2 = threading.Thread(target=thread_fn, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["A"] == "default-A"
        assert results["B"] == "default-B"
        # 主线程应该仍然是 None
        assert agent._get_default_session_id() is None

    def test_history_isolated_per_thread(self):
        """不同线程通过显式 set() 可以获得独立的消息历史。

        注意：ContextVar 的 default 参数返回的是同一个单例对象（mutable-default 陷阱）。
        要使线程间历史隔离，需要每个线程显式调用 _history_var.set() 创建自己的 deque。
        这也是 Agent 设计中引入 Session 机制的原因 —— Session 天然提供隔离的历史存储。
        """
        agent = _TestAgent()
        results = {}

        def thread_fn(thread_id: str):
            # 显式设置当前线程的独立历史（模拟 Session 提供隔离的方式）
            agent._history_var.set(deque(maxlen=agent.config.max_history_length))
            hist = agent._get_history()
            hist.append(Message(content=f"msg-from-{thread_id}", role="user"))
            results[thread_id] = len(hist)

        t1 = threading.Thread(target=thread_fn, args=("A",))
        t2 = threading.Thread(target=thread_fn, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 每个线程只看到自己的消息（1 条）
        assert results["A"] == 1
        assert results["B"] == 1

        # 主线程历史不受影响
        assert len(agent._get_history()) == 0

    def test_history_default_deque_is_shared(self):
        """验证 ContextVar 的 mutable-default 陷阱：未显式 set() 时所有线程共享同一个 deque。

        这解释了为什么 Agent 需要 Session 机制 —— 通过 Session 为每个对话提供隔离的历史存储，
        而非依赖 ContextVar 的默认值来存储可变状态。
        """
        agent = _TestAgent()
        results = {}

        def thread_fn(thread_id: str):
            # 不调用 set()，直接使用 default deque —— 所有线程共享
            hist = agent._get_history()
            hist.append(Message(content=f"msg-from-{thread_id}", role="user"))
            results[thread_id] = len(hist)

        t1 = threading.Thread(target=thread_fn, args=("A",))
        t2 = threading.Thread(target=thread_fn, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 由于共享同一个 default deque，两个线程的消息都在里面
        assert results["A"] >= 1
        assert results["B"] >= 2  # 线程 A 追加后再追加，所以 >= 2

    def test_contextvar_copies_independent_after_run(self):
        """验证 contextvars 的 copy_context 行为：每个线程的 ContextVar 修改互不干扰。"""
        agent = _TestAgent()

        # 主线程设置
        agent._set_default_session_id("main-default")
        assert agent._get_default_session_id() == "main-default"

        # 子线程修改不会影响主线程
        def child_fn():
            agent._set_default_session_id("child-default")
            assert agent._get_default_session_id() == "child-default"

        t = threading.Thread(target=child_fn)
        t.start()
        t.join()

        # 主线程的值不变
        assert agent._get_default_session_id() == "main-default"


# ---------------------------------------------------------------------------
# 测试：并发 run()
# ---------------------------------------------------------------------------

class TestConcurrentRun:
    """验证多线程并发调用 agent.run() 的安全性。"""

    def test_concurrent_run_different_sessions(self):
        """多线程使用不同 session_id 并发 run()，互不干扰。"""
        agent = _TestAgent()
        errors = []
        results = {}
        barrier = threading.Barrier(4, timeout=5)

        def thread_fn(thread_id: str):
            try:
                barrier.wait()  # 让所有线程同时开始
                msg = agent.run(f"hello from {thread_id}", session_id=f"sid-{thread_id}")
                results[thread_id] = msg.content
                # 验证该线程的 active session
                active = agent._get_active_session()
                results[f"{thread_id}_active"] = active.session_id if active else None
            except Exception as e:
                errors.append(f"{thread_id}: {e}")

        threads = [
            threading.Thread(target=thread_fn, args=(f"T{i}",))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        # 每个线程都得到了响应
        for i in range(4):
            assert f"T{i}" in results, f"Thread T{i} missing result"
            assert results[f"T{i}"] == "hello"

    def test_concurrent_run_no_session_id(self):
        """多线程不指定 session_id 并发 run()，各自自动创建默认 session。

        每个线程的 active session 和 default session ID 通过 ContextVar 隔离，
        但消息历史存储在各自的 Session 对象中（而非 ContextVar 的默认 deque），
        因此每个线程拥有完全独立的消息历史。
        """
        agent = _TestAgent()
        errors = []
        results = {}
        barrier = threading.Barrier(4, timeout=5)

        def thread_fn(idx: int):
            try:
                barrier.wait()
                msg = agent.run(f"hello from thread {idx}")
                active = agent._get_active_session()
                results[idx] = {
                    "response": msg.content,
                    "active_session": active.session_id if active else None,
                    "default_session": agent._get_default_session_id(),
                    # 消息历史存储在 Session 中（隔离的），而非 ContextVar 的默认 deque
                    "session_history_len": len(active.get_history()) if active else 0,
                }
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"

        # 每个线程都应有独立的 session 历史（本轮对话的 user + assistant = 2 条）
        for i in range(4):
            assert results[i]["response"] == "hello"
            assert results[i]["active_session"] is not None
            assert results[i]["default_session"] is not None
            assert results[i]["session_history_len"] == 2, \
                f"Thread {i} session_history_len={results[i]['session_history_len']}, expected 2"

        # 验证不同线程获得了不同的 session
        session_ids = {r["active_session"] for r in results.values()}
        assert len(session_ids) == 4, \
            f"Expected 4 distinct sessions, got {len(session_ids)}"

    def test_concurrent_run_same_session(self):
        """多线程使用同一个 session_id 并发 run()，共享 session 状态。"""
        agent = _TestAgent()
        agent.create_session("shared-session")
        barrier = threading.Barrier(3, timeout=5)
        errors = []

        def thread_fn(idx: int):
            try:
                barrier.wait()
                agent.run(f"msg-{idx}", session_id="shared-session")
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # 不应该崩溃（同一 session 并发 run 不是推荐用法，但不应该抛异常）
        assert len(errors) == 0, f"Errors: {errors}"

        # shared session 的消息历史应包含所有线程的消息
        hist = agent.get_history("shared-session")
        assert len(hist) >= 3 * 2  # 每个线程至少 2 条消息


# ---------------------------------------------------------------------------
# 测试：Session 池并发安全
# ---------------------------------------------------------------------------

class TestSessionPoolConcurrency:
    """验证 _sessions 字典在并发操作下的安全性（RLock 保护）。"""

    def test_concurrent_create_sessions(self):
        """并发创建大量 session 不会丢失或损坏。"""
        agent = _TestAgent()
        num_threads = 20
        errors = []

        def thread_fn(idx: int):
            try:
                sid = agent.create_session(f"test-session-{idx}")
                # 立即验证可以获取到
                sess = agent.get_session(sid)
                if sess is None:
                    errors.append(f"Thread {idx}: created session but get_session returned None")
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors: {errors}"
        sessions = agent.list_sessions()
        assert len(sessions) == num_threads, \
            f"Expected {num_threads} sessions, got {len(sessions)}"

    def test_concurrent_create_and_destroy(self):
        """并发创建和销毁 session 不会导致数据竞争。"""
        agent = _TestAgent()
        errors = []
        num_ops = 30

        def creator(idx: int):
            try:
                agent.create_session(f"cd-session-{idx}")
            except Exception as e:
                errors.append(f"Creator {idx}: {e}")

        def destroyer():
            try:
                sessions = agent.list_sessions()
                for sid in sessions:
                    agent.destroy_session(sid)
            except Exception as e:
                errors.append(f"Destroyer: {e}")

        # 先创建一些 session
        for i in range(num_ops):
            agent.create_session(f"cd-session-{i}")

        # 并发创建和销毁
        threads = []
        for i in range(num_ops, num_ops + 10):
            threads.append(threading.Thread(target=creator, args=(i,)))
        for _ in range(5):
            threads.append(threading.Thread(target=destroyer))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_list_sessions_safe(self):
        """并发调用 list_sessions 和 create_session 不会抛出异常。"""
        agent = _TestAgent()
        errors = []
        barrier = threading.Barrier(10, timeout=5)

        def worker():
            try:
                barrier.wait()
                for _ in range(50):
                    # 随机执行不同操作
                    agent.create_session()
                    _ = agent.list_sessions()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_get_session_thread_safety(self):
        """并发 get_session 不会因并发修改而崩溃。"""
        agent = _TestAgent()
        sid = agent.create_session("shared")

        errors = []

        def reader():
            for _ in range(100):
                try:
                    s = agent.get_session("shared")
                    if s is not None:
                        _ = s.session_id
                except Exception as e:
                    errors.append(str(e))

        def writer():
            for i in range(50):
                try:
                    agent.create_session(f"extra-{i}")
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"


# ---------------------------------------------------------------------------
# 测试：History 线程隔离
# ---------------------------------------------------------------------------

class TestHistoryIsolation:
    """验证消息历史在不同线程间的隔离性。"""

    def test_history_not_shared_across_threads_without_session(self):
        """不指定 session_id 时，不同线程的历史彼此隔离。

        由于 ContextVar default 是共享的（mutable-default 陷阱），直接使用 _get_history()
        会共享 deque。正确的隔离方式是通过 Session 机制：每个线程创建自己的 session，
        消息历史通过 session 存储，天然互不干扰。
        """
        agent = _TestAgent()
        histories = {}

        def thread_fn(thread_id: str):
            # 正确做法：每个线程使用独立的 session
            sid = agent.create_session(f"iso-{thread_id}")
            agent.switch_session(sid)
            agent.add_message(Message(content=f"from-{thread_id}", role="user"))
            active = agent._get_active_session()
            histories[thread_id] = [m.content for m in active.get_history()] if active else []

        t1 = threading.Thread(target=thread_fn, args=("A",))
        t2 = threading.Thread(target=thread_fn, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        content_a = histories.get("A", [])
        content_b = histories.get("B", [])
        assert "from-A" in content_a
        assert "from-B" not in content_a
        assert "from-B" in content_b
        assert "from-A" not in content_b

    def test_history_shared_via_session(self):
        """通过同一个 session_id，不同线程可以看到共享的历史。"""
        agent = _TestAgent()
        sid = agent.create_session("shared-hist")

        agent.add_message(Message(content="pre-shared", role="user"), session_id=sid)

        def thread_fn(msg: str):
            agent.add_message(Message(content=msg, role="user"), session_id=sid)

        t1 = threading.Thread(target=thread_fn, args=("from-t1",))
        t2 = threading.Thread(target=thread_fn, args=("from-t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        hist = agent.get_history(session_id=sid)
        contents = [m.content for m in hist]
        assert "pre-shared" in contents
        # 两条线程的消息都在 session 历史中
        assert "from-t1" in contents
        assert "from-t2" in contents


# ---------------------------------------------------------------------------
# 测试：Session 切换并发安全
# ---------------------------------------------------------------------------

class TestSessionSwitchConcurrency:
    """验证 session 切换在并发环境下的安全性。"""

    def test_concurrent_switch_session(self):
        """并发切换 session 不会导致状态错乱（每个线程的 active session 独立）。"""
        agent = _TestAgent()
        agent.create_session("sess-A")
        agent.create_session("sess-B")
        agent.create_session("sess-C")
        results = {}

        barrier = threading.Barrier(3, timeout=5)

        def thread_fn(target_sid: str):
            try:
                barrier.wait()
                agent.switch_session(target_sid)
                active = agent._get_active_session()
                results[target_sid] = active.session_id if active else None
            except Exception as e:
                results[f"error-{target_sid}"] = f"{type(e).__name__}: {e}"

        threads = [
            threading.Thread(target=thread_fn, args=(sid,))
            for sid in ["sess-A", "sess-B", "sess-C"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # 每个线程都应成功切换到目标 session
        assert results.get("sess-A") == "sess-A", f"sess-A result: {results.get('sess-A')}"
        assert results.get("sess-B") == "sess-B", f"sess-B result: {results.get('sess-B')}"
        assert results.get("sess-C") == "sess-C", f"sess-C result: {results.get('sess-C')}"

        # 主线程的 active session 未变（未被切换影响）
        assert agent._get_active_session() is None


# ---------------------------------------------------------------------------
# 测试：工具状态 ContextVar 线程隔离
# ---------------------------------------------------------------------------

class StatefulMockTool(Tool):
    """有状态的 mock 工具，用于测试工具状态线程隔离。"""

    def __init__(self, name: str = "stateful_tool"):
        super().__init__(name=name, description="A stateful tool for thread safety testing")

    def _make_default_state(self) -> dict:
        return {"counter": 0}

    def run(self, parameters: dict, tool_call_id: str) -> Message:
        state = self._get_state()
        state["counter"] += 1
        return Message(
            content=f"counter={state['counter']}",
            role="tool",
            tool_call_id=tool_call_id
        )

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={"input": ToolProperty(type="string", description="input")},
            required=[]
        )

    def increment(self) -> int:
        state = self._get_state()
        state["counter"] += 1
        return state["counter"]

    def get_counter(self) -> int:
        return self._get_state()["counter"]


class TestToolStateIsolation:
    """验证工具 ContextVar 状态在多线程中的隔离性。"""

    def test_tool_state_isolated_per_thread(self):
        """每个线程拥有独立的工具状态副本。"""
        tool = StatefulMockTool()
        results = {}
        barrier = threading.Barrier(3, timeout=5)

        def thread_fn(thread_id: str):
            barrier.wait()
            # 每个线程做不同次数的 increment
            increments = {"A": 3, "B": 5, "C": 1}[thread_id]
            for _ in range(increments):
                tool.increment()
            results[thread_id] = tool.get_counter()

        threads = [
            threading.Thread(target=thread_fn, args=(tid,))
            for tid in ["A", "B", "C"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert results["A"] == 3
        assert results["B"] == 5
        assert results["C"] == 1
        # 主线程计数器未受影响
        assert tool.get_counter() == 0

    def test_tool_state_reset_per_thread(self):
        """reset() 只影响当前线程的工具状态。"""
        tool = StatefulMockTool()

        # 在主线程中递增
        tool.increment()
        tool.increment()
        assert tool.get_counter() == 2

        # 在子线程中递增
        child_counter = []

        def child_fn():
            tool.increment()
            child_counter.append(tool.get_counter())
            tool.reset()
            child_counter.append(tool.get_counter())

        t = threading.Thread(target=child_fn)
        t.start()
        t.join()

        # 子线程中的操作
        assert child_counter == [1, 0]
        # 主线程状态不变
        assert tool.get_counter() == 2

    def test_tool_state_save_restore_per_thread(self):
        """save/restore 工具状态仅限当前线程。"""
        tool = StatefulMockTool()

        # 主线程设置状态
        tool.increment()
        tool.increment()
        saved = tool.get_session_state()
        assert saved["counter"] == 2

        # 子线程做不同的操作
        def child_fn():
            tool.increment()
            tool.increment()
            tool.increment()
            assert tool.get_counter() == 3
            # restore 主线程的快照
            tool.restore_session_state(saved)
            assert tool.get_counter() == 2

        t = threading.Thread(target=child_fn)
        t.start()
        t.join()

        # 主线程状态不变
        assert tool.get_counter() == 2


# ---------------------------------------------------------------------------
# 测试：concurrent.futures 线程池
# ---------------------------------------------------------------------------

class TestThreadPoolSafety:
    """验证在使用 ThreadPoolExecutor 时的线程安全性。"""

    def test_thread_pool_executor_run(self):
        """ThreadPoolExecutor 中并发调用 agent.run() 安全。"""
        import concurrent.futures
        agent = _TestAgent()
        errors = []

        def task(idx: int) -> str:
            try:
                msg = agent.run(f"task {idx}", session_id=f"pool-session-{idx}")
                return msg.content
            except Exception as e:
                errors.append(f"Task {idx}: {e}")
                return f"error: {e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(task, i) for i in range(12)]
            results = [f.result(timeout=10) for f in futures]

        assert len(errors) == 0, f"Errors: {errors}"
        assert all(r == "hello" for r in results), f"Unexpected results: {results}"

        # 所有 session 都已创建
        sessions = agent.list_sessions()
        assert len(sessions) == 12

    def test_thread_pool_mixed_operations(self):
        """线程池中混合执行 create/run/switch/destroy 操作安全。"""
        import concurrent.futures
        agent = _TestAgent()
        errors = []

        def create_and_run(idx: int):
            try:
                sid = agent.create_session(f"mixed-{idx}")
                msg = agent.run(f"hello {idx}", session_id=sid)
                return msg.content
            except Exception as e:
                errors.append(f"Task {idx}: {e}")
                return None

        def list_and_destroy():
            try:
                sessions = agent.list_sessions()
                # 销毁最旧的几个
                for sid in sessions[:3]:
                    agent.destroy_session(sid)
            except Exception as e:
                errors.append(f"Cleaner: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(20):
                futures.append(executor.submit(create_and_run, i))
            for _ in range(3):
                futures.append(executor.submit(list_and_destroy))
            for f in futures:
                f.result(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"


# ---------------------------------------------------------------------------
# 测试：ContextVar 的 copy_context 行为
# ---------------------------------------------------------------------------

class TestContextVarCopy:
    """验证 contextvars.copy_context() 在 Agent 中的行为。"""

    def test_copy_context_captures_isolated_state(self):
        """copy_context() 捕获的 ContextVar 状态是独立的快照。"""
        agent = _TestAgent()

        def child_fn(result_holder: list):
            agent._set_default_session_id("child")
            agent._set_active_session(None)  # child context
            result_holder.append(agent._get_default_session_id())

        # 在主线程设置
        agent._set_default_session_id("parent")

        # 使用 copy_context 在子线程运行
        ctx = contextvars.copy_context()
        result = []

        t = threading.Thread(target=lambda: ctx.run(child_fn, result))
        t.start()
        t.join()

        # 子线程的操作不影响主线程
        assert result == ["child"]
        assert agent._get_default_session_id() == "parent"
