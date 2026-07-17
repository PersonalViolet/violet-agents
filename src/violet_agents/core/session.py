"""会话隔离模块 - 将每个对话的可变状态封装为独立的 Session 对象"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
from datetime import datetime, timezone
import uuid
import copy

from .message import Message

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Agent

@dataclass
class Session:
    """一个隔离的会话，持有单个对话的全部可变状态。

    Agent 实例持有不变的配置（LLM、ToolRegistry、system_prompt），
    Session 持有每个对话独立的状态（历史、工具状态快照、临时工具、钩子、轮次）。
    
    该类的agent_state字段提供了一个通用的状态存储空间，供Agent实现类自由使用，可以存储任何与当前对话相关的状态信息，而不需要修改Session类的定义。

    生命周期:
        1. agent.create_session() 创建
        2. agent.run("input", session_id=...) 使用
        3. agent.destroy_session(session_id) 销毁
        4. save_session() / restore_session() 持久化
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    max_history_length: int = 100

    # --- 消息历史 ---
    _history: deque = field(default_factory=deque)

    # --- 元数据 ---
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Agent 专用可变状态（agent 实现类可自由存储状态） ---
    agent_state: Dict[str, Any] = field(default_factory=dict)

    # --- 每会话工具状态快照 (tool_name -> state_dict) ---
    _tool_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- 钩子 (hook_name -> list of callbacks) ---
    hooks: Dict[str, List[Callable]] = field(default_factory=dict)

    # --- ReactAgent 专用字段 ---
    # temp_tools: List[Dict[str, Any]] = field(default_factory=list)
    # temp_tools_names: Set[str] = field(default_factory=set)
    # temp_tools_last_call_round: Dict[str, int] = field(default_factory=dict)
    # current_round: int = 0

    def __post_init__(self):
        """确保 _history 的 maxlen 正确，初始化 hooks 结构。"""
        history = self._history
        self._history = deque(history, maxlen=self.max_history_length if self.max_history_length > 0 else None)
        if not self.hooks:
            self.hooks = {"UserPromptSubmit": [], "PreToolCall": [], "PostToolCall": []}

    # --- 历史方法 ---

    def add_message(self, message: Message) -> None:
        """将消息添加到历史中，并更新 updated_at 时间戳。
        
        Args:
            message (Message): 要添加的消息对象
        """
        self._history.append(message)
        self._touch()

    def get_history(self) -> deque:
        """获取历史消息的副本，避免外部修改原始历史。"""
        return self._history.copy()

    def clear_history(self) -> None:
        """清空历史消息，并更新 updated_at 时间戳。"""
        self._history.clear()
        self._touch()

    # --- 工具状态管理 ---

    def save_tool_state(self, tool_name: str, state: Dict[str, Any]) -> None:
        """保存某个工具的当前可变状态快照。"""
        self._tool_state[tool_name] = copy.deepcopy(state)

    def get_tool_state_ref(self, tool_name: str) -> Dict[str, Any]:
        """获取某个工具的上次保存状态，不存在则返回空 dict。"""
        return self._tool_state.get(tool_name, {})

    def get_tool_state(self, tool_name: str) -> Dict[str, Any]:
        """获取某个工具的上次保存状态的副本，不存在则返回空 dict。"""
        return copy.deepcopy(self._tool_state.get(tool_name, {}))

    # --- 生命周期 ---

    def _touch(self) -> None:
        """当前 session 被访问或修改时更新 updated_at 时间戳。"""
        self.updated_at = datetime.now(timezone.utc)

    def is_expired(self, ttl_seconds: Optional[int]) -> bool:
        if ttl_seconds is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.updated_at).total_seconds()
        return elapsed > ttl_seconds

    # --- 序列化 ---

    def to_dict(self) -> Dict[str, Any]:
        """将 session 序列化为 JSON 安全的 dict（钩子回调不序列化）。"""
        return {
            "session_id": self.session_id,
            "max_history_length": self.max_history_length,
            "history": [msg.model_dump() for msg in self._history],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tool_state": self._tool_state,
            "agent_state": self.agent_state,
            # "temp_tools": self.temp_tools,
            # "temp_tools_names": list(self.temp_tools_names),
            # "temp_tools_last_call_round": self.temp_tools_last_call_round,
            # "current_round": self.current_round,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """从 to_dict() 产生的 dict 恢复 session。钩子需单独重新注册。"""
        # 深拷贝data，防止在恢复过程中修改原始数据导致副作用
        data = copy.deepcopy(data)

        session = cls(
            session_id=data["session_id"],
            max_history_length=data.get("max_history_length", 100),
            metadata=data.get("metadata", {}),
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session._history = deque(
            [Message(**msg) for msg in data.get("history", [])],
            maxlen=session.max_history_length if session.max_history_length > 0 else None,
        )
        session._tool_state = copy.deepcopy(data.get("tool_state", {}))
        session.agent_state = copy.deepcopy(data.get("agent_state", {}))
        # session.temp_tools = data.get("temp_tools", [])
        # session.temp_tools_names = set(data.get("temp_tools_names", []))
        # session.temp_tools_last_call_round = data.get("temp_tools_last_call_round", {})
        # session.current_round = data.get("current_round", 0)
        return session


class _SessionContext:
    """Agent.session() 上下文管理器的内部实现。"""

    def __init__(self, agent: "Agent", session_id: str):
        self.agent = agent
        self.session_id = session_id
        self._previous_session = None

    def __enter__(self):
        current = self.agent._get_active_session()
        self._previous_session = current.session_id if current else None
        self.agent._resolve_session(self.session_id)  # 确保 session 存在
        self.agent.switch_session(self.session_id)
        return self.agent._get_active_session()

    def __exit__(self, *args):
        if self._previous_session:
            self.agent.switch_session(self._previous_session)
        else:
            self.agent.deactivate_session()
