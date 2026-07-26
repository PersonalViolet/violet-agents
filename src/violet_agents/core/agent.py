"""Agent基类"""
import asyncio
import concurrent.futures
import functools
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, overload, Callable, Literal, Set, TypeAlias, Annotated, Tuple, Union
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
from .llm import VioletAgentsLLM
from .config import Config
from .message import Message
from collections import deque
import os
import uuid
import contextvars
import threading
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .session import Session
    from ..tools.registry import ToolRegistry
    from ..tools import Tool
class Agent(ABC):

    """Agent基类

    Agent 持有不变配置（LLM、system_prompt、config），Session 持有可变状态。
    一个 Agent 可创建多个独立 Session，每个 Session 拥有完全隔离的消息历史、工具状态和对话上下文。

    该类设计的钩子系统有两个层级：
    1. Agent级别的默认钩子
    2. Session级别的钩子（覆盖默认钩子）

    Agent基类支持的钩子事件，实现类可以直接注册回调函数，在事件发生时会自动触发回调。
    实现类也可以自定义更多钩子事件，自定义事件的触发时机由实现类控制。

    目前内置的钩子事件包括：
    - SessionInit: 在每次创建新 Session 时触发，回调参数为新创建的 Session 对象。适合在这里为 Session 注入默认钩子。
    - SessionSwitch: 在agent实例真正切换 Session 时触发，回调参数为一个元组 (old_session, new_session)。适合在这里处理切换时的状态保存/恢复等逻辑。

    线程安全说明：
    使用 contextvars 实现每个执行上下文（线程/asyncio task）独立的 active session 和历史记录，
    多线程并发调用 agent.run() 互不干扰。Session 池 (_sessions) 通过 RLock 保护并发读写。
    """

    def __init__(self,
                 name: str,
                 llm: VioletAgentsLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None,
                 tool_registry: Optional["ToolRegistry"] = None,
                 max_workers: int = 5
                 ):
        self.name = name
        self.llm = llm or VioletAgentsLLM()
        self.system_prompt = system_prompt
        self.config = config or Config()
        if tool_registry is None:
            from ..tools.registry import ToolRegistry
            tool_registry = ToolRegistry()
        self.tool_registry = tool_registry

        self._tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self._agent_hooks: Dict[str, List[Callable]] = {
            "SessionInit": [],
            "PreSessionSwitch": [],
            "PostSessionSwitch": [],
        }
        self.registry_agent_hook("SessionInit", self._init_session_hooks)

        self._default_session_hooks: Dict[str, List[Callable]] = {
            "TurnStart": [],
            "TurnEnd": [],
        }
        # --- ContextVar：每执行上下文的隔离状态 ---
        self._active_session_var: contextvars.ContextVar[Optional["Session"]] = \
            contextvars.ContextVar(f"_active_session_{id(self)}", default=None)
        self._history_var: contextvars.ContextVar[deque[Message]] = \
            contextvars.ContextVar(
                f"_history_{id(self)}",
                default=deque[Message](
                    maxlen=self.config.max_history_length if self.config.max_history_length > 0 else None
                )
            )
        self._default_session_id_var: contextvars.ContextVar[Optional[str]] = \
            contextvars.ContextVar(f"_default_session_id_{id(self)}", default=None)

        # --- Session 管理 ---
        self._sessions: Dict[str, "Session"] = {}   # session_id -> Session 对象
        self._sessions_lock = threading.RLock()      # Session 池并发保护

        self.register_session_hook("TurnStart", self._restore_all_tool_states_hook)
        self.register_session_hook("TurnEnd", self._save_all_tool_states_hook)

    # --- ContextVar 辅助方法 ---

    def _get_active_session(self) -> Optional["Session"]:
        """获取当前执行上下文的活跃 Session。"""
        return self._active_session_var.get()

    def _set_active_session(self, sess: Optional["Session"]) -> None:
        """设置当前执行上下文的活跃 Session。"""
        self._active_session_var.set(sess)

    def _get_history(self) -> deque[Message]:
        """获取当前执行上下文的默认消息历史（向后兼容）。"""
        return self._history_var.get()

    def _get_default_session_id(self) -> Optional[str]:
        """获取当前执行上下文的默认 session ID。"""
        return self._default_session_id_var.get()

    def _set_default_session_id(self, sid: Optional[str]) -> None:
        """设置当前执行上下文的默认 session ID。"""
        self._default_session_id_var.set(sid)

    # --- 主运行接口 ---

    def run(self, input_text: str, session_id: Optional[str] = None, **kwargs) -> Message:
        """运行Agent，处理输入并返回任务完成后的结果。

        该方法不会修改session相关属性，负责解析 session_id，切换到对应的 session，触发 TurnStart/TurnEnd 钩子，并调用 do_run 处理输入。实现类只需关注 do_run 的实现即可。

        该方法会修改的session属性如下：
        - _tool_state: 访问工具状态以便在 do_run 中使用工具时能够正确保存/恢复状态。
        - 其他属性（如消息历史）由 do_run 实现类根据需要修改。
    
            设计目标是提供一个清晰的运行流程和钩子触发机制，让实现类专注于输入处理和工具调用的核心逻辑
        Args:
            input_text: 用户输入文本
            session_id: 可选，指定使用的 session ID。不存在则自动创建。
        """
        sess = self._resolve_session(session_id)
        self.switch_session(sess.session_id)
        self._trigger_session_hooks("TurnStart", input_text, sess=sess)
        response = self.do_run(input_text, session=sess, **kwargs)
        self._trigger_session_hooks("TurnEnd", response, sess=sess)
        return response


    @abstractmethod
    def do_run(self, 
               input_text: str, 
               session: "Session", 
               **kwargs) -> Message:
        """运行Agent，处理输入并返回任务完成后的结果。

        Args:
            input_text: 用户输入文本
            session: run方法传入的 Session 对象，已由 run 方法解析获得并自动切换到该 session。实现类直接使用该参数即可。
        """
    

    async def arun(self, input_text: str, session_id: Optional[str] = None, **kwargs) -> Message:
        """异步运行Agent，处理输入并返回任务完成后的结果。

        该方法会在后台线程中调用同步的 run 方法，适用于异步环境下的调用。
        实现类无需重写该方法，只需实现 do_run 即可。
        子类可以覆盖此方法实现更复杂的异步逻辑。

        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        func = functools.partial(self.run, input_text, session_id=session_id, **kwargs)
        return await loop.run_in_executor(None, lambda: ctx.run(func))
    

    def register_tool(self, tool: "Tool", is_defer: bool = False) -> "Agent":
        """注册工具到 该Agent。支持链式调用。

        Args:
            tool: 要注册的工具实例
            is_defer: 是否注册为延迟工具。默认 False。
        Returns:
            self: 该Agent实例，支持链式调用。
        """
        self.tool_registry.register_tool(tool, is_defer=is_defer)
        return self

    def register_dynamic_tool(self, 
                              tools_source: Dict[str, "Tool"],
                              is_defer: bool = False) -> "Agent":
        """注册动态工具到 该Agent。支持链式调用。
        
        Args:
            tools_source: 包含工具实例的字典，键为工具名称，值为工具实例。
            is_defer: 是否注册为延迟工具。默认 False。
        Returns:
            self: 该Agent实例，支持链式调用。
        """
        self.tool_registry.register_dynamic_tools(tools_source, is_defer=is_defer)
        return self
    
    def execute_tool(self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]) -> Message:
        """
        执行工具调用。该方法会将工具调用请求传递给 ToolRegistry 进行处理。
        子类无需重写该方法，除非需要自定义工具调用的处理逻辑
        """
        if self.tool_registry is None:
            raise RuntimeError("ToolRegistry is not initialized.")
        return self.tool_registry.execute_tool(tool_call)

    def execute_tools_concurrently(self, tool_calls: List[Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]]) -> List[Message]:
        """批量并发执行工具调用（同步），结果顺序与输入顺序一致。

        与 ToolRegistry.execute_tools_concurrently() 的关键区别：
        本方法通过 self.execute_tool() 执行每个工具调用，因此子类（如 ReactAgent）重写
        execute_tool() 后注入的 PreToolCall / PostToolCall 等钩子也会对并发路径生效。
        """
        if not tool_calls:
            return []

        # 使用 Agent 自身的线程池提交任务，确保 self.execute_tool() 被子类正确分发
        future_to_index: dict[concurrent.futures.Future, int] = {}
        for i, tool_call in enumerate(tool_calls):
            ctx = contextvars.copy_context()
            future = self._tool_executor.submit(ctx.run, self.execute_tool, tool_call)
            future_to_index[future] = i

        results: list[Optional[Message]] = [None] * len(tool_calls)
        exceptions: list[tuple[int, Exception]] = []

        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                exceptions.append((idx, e))

        if exceptions:
            failed_names = [
                f"tool_calls[{i}]: {type(e).__name__}: {e}"
                for i, e in exceptions
            ]
            raise RuntimeError(
                f"{len(exceptions)}/{len(tool_calls)} 个工具执行失败:\n" +
                "\n".join(failed_names)
            )

        return results  # type: ignore[return-value]
    
    # --- 历史方法（委托模式） ---
    def add_message(self,
                    message: Message,
                    session_id: Optional[str] = None):
        """添加消息到历史。指定 session_id 则添加到那个 session，否则添加到 active session 或默认历史（向后兼容）。"""
        if session_id:
            sess = self.get_session(session_id)
            if sess is None:
                raise KeyError(f"Session '{session_id}' not found")
            sess.add_message(message)
            return
        active = self._get_active_session()
        if active:
            active.add_message(message)
        else:
            self._get_history().append(message)

    def get_history(self, session_id: Optional[str] = None) -> deque[Message]:
        """获取消息历史。指定 session_id 则获取那个 session 的历史，否则获取 active session 的消息历史或默认历史（向后兼容）。

        Args:
            session_id: 可选，指定要获取历史的 session ID。不存在则获取 active session 的消息历史或默认历史。
        """
        if session_id:
            sess = self.get_session(session_id)
            if sess is None:
                raise KeyError(f"Session '{session_id}' not found")
            return sess.get_history()
        active = self._get_active_session()
        if active:
            return active.get_history()
        return self._get_history().copy()

    def clear_history(self, session_id: Optional[str] = None):
        """清理消息历史。指定 session_id 则清理那个 session 的历史，否则清理 active session 或默认历史（向后兼容）。

        Args:
            session_id: 可选，指定要清理历史的 session ID。不存在则清理 active session 或默认历史。
        """
        if session_id:
            sess = self.get_session(session_id)
            if sess is None:
                raise KeyError(f"Session '{session_id}' not found")
            sess.clear_history()
            return
        active = self._get_active_session()
        if active:
            active.clear_history()
        else:
            self._get_history().clear()

    # --- Session 生命周期 ---
    def create_session(self, session_id: Optional[str] = None, **kwargs) -> str:
        """创建新 session，返回 session_id。不自动激活。若未指定 session_id，则自动生成"""
        from .session import Session
        max_history = self.config.max_history_length
        sess = Session(
            session_id=session_id or uuid.uuid4().hex[:12],
            max_history_length=max_history,
            **kwargs
        )
        self._trigger_agent_hooks("SessionInit", sess)
        with self._sessions_lock:
            self._sessions[sess.session_id] = sess
        if self.config.auto_cleanup_sessions:
            self._cleanup_expired_sessions()
        return sess.session_id

    def get_session(self, session_id: str) -> Optional["Session"]:
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def destroy_session(self, session_id: str) -> bool:
        """永久删除一个 session。如果是 active session 则先保存工具状态并清除。"""
        with self._sessions_lock:
            if session_id not in self._sessions:
                return False
            active = self._get_active_session()
            if active and active.session_id == session_id:
                self._save_all_tool_states()
                self._set_active_session(None)
            del self._sessions[session_id]
        return True

    def list_sessions(self) -> List[str]:
        with self._sessions_lock:
            return list(self._sessions.keys())

    def switch_session(self, session_id: str) -> None:
        """
        切换到指定 session，如果该 session 已经是当前 active session 则不执行任何操作，也不会触发Pre/PostSessionSwitch钩子。
        """
        sess = self.get_session(session_id)
        if sess is None:
            raise KeyError(f"Session '{session_id}' not found")
        if self._is_active_session(sess):
            return

        previous_session = self._get_active_session()
        if previous_session is not None:
            self._save_all_tool_states()

        self._trigger_agent_hooks("PreSessionSwitch", (previous_session, sess))
        self._activate_session(sess)
        self._restore_all_tool_states()
        self._trigger_agent_hooks("PostSessionSwitch", (previous_session, sess))

    def _is_active_session(self, sess: Union["Session", str]) -> bool:
        """判断给定 session 是否为当前 active session。参数可以是 session_id 或 Session 对象。"""
        active = self._get_active_session()
        if isinstance(sess, str):
            return active is not None and active.session_id == sess
        else:
            return active is not None and active.session_id == sess.session_id

    def _activate_session(self, sess: "Session") -> None:
        """
        激活 session"""
        self._set_active_session(sess)

    def session(self, session_id: str):
        """上下文管理器：在指定 session 内运行代码块。

        Usage:
            with agent.session(sid):
                agent.run("hello")
        """
        from .session import _SessionContext
        return _SessionContext(self, session_id)

    def save_session(self, session_id: str) -> Dict[str, Any]:
        """将 session 序列化为 JSON 安全的 dict。"""
        with self._sessions_lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"Session '{session_id}' not found")
        active = self._get_active_session()
        if active and active.session_id == session_id:
            self._save_all_tool_states()
        return sess.to_dict()
    
    def restore_session(self, data: Dict[str, Any]) -> str:
        """从 save_session() 产生的 dict 恢复 session。返回 session_id。"""
        from .session import Session
        sess = Session.from_dict(data)
        self._init_session_hooks(sess)
        with self._sessions_lock:
            self._sessions[sess.session_id] = sess
        return sess.session_id

    # --- 工具状态保存/恢复的内部实现 ---
    def _save_all_tool_states(self) -> None:
        """
        保存当前 active session 中所有工具的状态到 session 的 tool_state 中。工具状态以 JSON 安全的格式保存，具体内容由工具的 get_session_state() 方法定义。

        工具实例的可变状态会重置

        子类可重写"""
        sess = self._get_active_session()
        if not sess:
            return
        for tool_name, tool in self.tool_registry.get_all_tools().items():
            sess.save_tool_state(tool_name, tool.get_session_state())

    def _restore_all_tool_states(self) -> None:
        """
        从当前 active session 中恢复所有工具的状态。工具状态以 JSON 安全的格式保存，具体内容由工具的 get_session_state() 方法定义。

        子类可重写
        """
        sess = self._get_active_session()
        if not sess:
            return
        for tool_name, tool in self.tool_registry.get_all_tools().items():
            state = sess.get_tool_state(tool_name)
            tool.restore_session_state(state)


    # --- Session 解析辅助 ---
    def _resolve_session(self, session_id: Optional[str]) -> "Session":
        """解析本次 run 使用的 session。
            - 如果 session_id 存在且有效，使用指定 session。
            - 否则如果有 active session，使用 active session。
            - 否则自动创建默认 session（仅第一次调用时创建）。
            - 兼容旧版历史：如果没有 active session，消息历史仍保存在 ContextVar 的默认 history 中。
            - 注意：session 切换会保存当前 session 的工具状态，但不会自动清理消息历史。子类可重写 _save/_restore_tool_states_impl 来实现更复杂的状态管理。
            - 设计目标是提供灵活的 session 管理，同时保持向后兼容和简单易用的接口。

            Returns:
                解析获得的 Session 对象
        """
        if session_id:
            with self._sessions_lock:
                if session_id in self._sessions:
                    return self._sessions[session_id]
            # Session 不存在，在锁外创建（create_session 内部会自行加锁）
            self.create_session(session_id=session_id)
            with self._sessions_lock:
                return self._sessions[session_id]
        active = self._get_active_session()
        if active:
            return active
        # 向后兼容：自动创建默认 session
        default_sid = self._get_default_session_id()
        if default_sid is None:
            default_sid = self.create_session()
            self._set_default_session_id(default_sid)
        with self._sessions_lock:
            return self._sessions[default_sid]

    def deactivate_session(self) -> None:
        """运行结束后保存工具状态。清理 active session"""
        self._save_all_tool_states()
        self._set_active_session(None)

    def _cleanup_expired_sessions(self) -> None:
        """根据 max_sessions 和 TTL 清理过期/超量 session。"""
        max_sessions = self.config.max_sessions
        ttl = self.config.session_default_ttl
        default_sid = self._get_default_session_id()
        active = self._get_active_session()
        if ttl is not None:
            with self._sessions_lock:
                expired = [sid for sid, s in self._sessions.items() if s.is_expired(ttl)]
            for sid in expired:
                if sid == default_sid:
                    continue
                self.destroy_session(sid)
        with self._sessions_lock:
            if max_sessions and len(self._sessions) > max_sessions:
                sorted_sessions = sorted(
                    [s for s in self._sessions.values()
                     if s.session_id != default_sid and s != active],
                    key=lambda s: s.updated_at,
                )
                to_remove = len(self._sessions) - max_sessions
                for sess in sorted_sessions[:to_remove]:
                    del self._sessions[sess.session_id]
    
    @overload
    def register_session_hook(self, event: Literal["TurnStart"],
                       callback: Callable[[str], None],
                       session_id: Optional[str] = None) -> None: 
        """注册 Session 级别的 TurnStart 钩子。回调参数是本轮对话的输入文本。"""
        ...

    @overload
    def register_session_hook(self, event: Literal["TurnEnd"],
                       callback: Callable[[Message], None],
                       session_id: Optional[str] = None) -> None: 
        """注册 Session 级别的 TurnEnd 钩子。回调参数是本轮对话的输出消息对象。"""
        ...

    def register_session_hook(self, event: str, callback: Callable,
                       session_id: Optional[str] = None):
        """注册Session级别的钩子。指定 session_id 则仅对那个 session 生效，否则作为默认钩子应用到所有新 session。"""
        if session_id:
            sess = self.get_session(session_id)
            if sess is None:
                raise KeyError(f"Session '{session_id}' not found")
            sess.hooks[event].append(callback)
        else:
            if event not in self._default_session_hooks:
                self._default_session_hooks[event] = []
            self._default_session_hooks[event].append(callback)
            with self._sessions_lock:
                sessions = list(self._sessions.values())
            for sess in sessions:
                sess.hooks[event].append(callback)

    @overload
    def registry_agent_hook(self, event: Literal["SessionInit"], callback: Callable[["Session"], None]) -> None: ...

    # @overload
    # def registry_agent_hook(self, event: Literal["SessionSwitch"], callback: Callable[["Session", "Session"], None]) -> None: ...

    @overload
    def registry_agent_hook(self, event: Literal["PreSessionSwitch"], callback: Callable[["Session", "Session"], None]) -> None: ...

    @overload
    def registry_agent_hook(self, event: Literal["PostSessionSwitch"], callback: Callable[["Session", "Session"], None]) -> None: ...


    def registry_agent_hook(self, event: str, callback: Callable):
        """注册 Agent 级别的钩子。该钩子不依赖于 session，直接绑定在 Agent 上。"""
        if event not in self._agent_hooks:
            self._agent_hooks[event] = []
        self._agent_hooks[event].append(callback)

    @overload
    def _trigger_session_hooks(self, event: Literal["TurnStart"], arg: str, sess: Optional["Session"] = None) -> None: 
        """
        触发 Session 级别的 TurnStart 钩子。参数 arg 是本轮对话的输入文本。
        """
        ...
    
    @overload
    def _trigger_session_hooks(self, event: Literal["TurnEnd"], arg: Message, sess: Optional["Session"] = None) -> None: 
        """
        触发 Session 级别的 TurnEnd 钩子。参数 arg 是本轮对话的输出消息对象。
        """
        ...

    def _trigger_session_hooks(self, event: str, *args, sess: Optional["Session"] = None):
        """触发钩子。

        1. session级别的钩子：若 sess 参数提供则仅触发那个 session 的钩子，否则触发 active session 的钩子。返回第一个非 None 的结果（如果有）。"""
        target = sess or self._get_active_session()
        if target is None:
            return None
        for callback in target.hooks.get(event, []):
            result = callback(*args)
            if result is not None:
                return result
        return None
    
    @overload
    def _trigger_agent_hooks(self, event: Literal["SessionInit"], arg: "Session") -> None: ...

    # @overload
    # def _trigger_agent_hooks(self, event: Literal["SessionSwitch"], arg: Tuple["Session", "Session"]) -> None: ...

    @overload
    def _trigger_agent_hooks(self, event: Literal["PreSessionSwitch"], arg: Tuple["Session", "Session"]) -> None: ...

    @overload
    def _trigger_agent_hooks(self, event: Literal["PostSessionSwitch"], arg: Tuple["Session", "Session"]) -> None: ...

    def _trigger_agent_hooks(self, event: str, *args):
        """触发 Agent 级别的钩子。返回第一个非 None 的结果（如果有）。"""
        for callback in self._agent_hooks.get(event, []):
            result = callback(*args)
            if result is not None:
                return result
        return None
    

    def _init_session_hooks(self, sess: "Session") -> None:
        """新 session 创建时，复制 agent 的session级别的默认钩子到session中。"""
        for event, callbacks in self._default_session_hooks.items():
            sess.hooks[event] = list(callbacks)

    def _restore_all_tool_states_hook(self, input_text: str) -> None:
        """TurnStart 钩子：在每轮对话开始时自动恢复工具状态到 session 中。"""
        self._restore_all_tool_states()

    def _save_all_tool_states_hook(self, msg: Message) -> None:
        """TurnEnd 钩子：在每轮对话结束时自动保存工具状态到 session 中。"""
        self._save_all_tool_states()
    



class SubAgent(Agent):
    """SubAgent基类 - 使用独立的环境变量配置 LLM"""

    def __init__(self,
                 name: str,
                 llm: Optional[VioletAgentsLLM] = None,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        api_key = os.getenv("SUB_AGENT_LLM_API_KEY")
        base_url = os.getenv("SUB_AGENT_LLM_BASE_URL")
        model = os.getenv("SUB_AGENT_LLM_MODEL")
        llm = llm or VioletAgentsLLM(api_key=api_key, base_url=base_url, model=model)
        super().__init__(name, llm, system_prompt, config)
