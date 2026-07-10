"""核心层 —— Agent 基类、LLM 接口、消息模型、配置与会话管理。"""

from .agent import Agent, SubAgent
from .llm import VioletAgentsLLM, SUPPORTED_PROVIDERS, TOOL_CHOICE
from .message import Message, MessageRole
from .config import Config
from .session import Session
from .exceptions import (
    VioletAgentException,
    LLMException,
    AgentException,
    ConfigException,
    ToolException,
)

__all__ = [
    # Agent
    "Agent",
    "SubAgent",
    # LLM
    "VioletAgentsLLM",
    "SUPPORTED_PROVIDERS",
    "TOOL_CHOICE",
    # 消息
    "Message",
    "MessageRole",
    # 配置与会话
    "Config",
    "Session",
    # 异常
    "VioletAgentException",
    "LLMException",
    "AgentException",
    "ConfigException",
    "ToolException",
]