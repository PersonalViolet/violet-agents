"""violet_agents — 基于 OpenAI 兼容 API 的轻量级 Python Agent 框架。

核心组件:
- Agent / SubAgent: Agent 基类与子 Agent
- VioletAgentsLLM: LLM 统一接口
- Config / Session: 配置与会话隔离
- Message / MessageRole: 消息模型
- Tool / ToolRegistry: 工具系统
- ReactAgent / SimpleAgent: 内置 Agent 实现

工具审批:
- ApprovalTool: 审批基类
- DefaultApprovalTool: 终端交互式审批
"""

# --- Agent 层 ---
from .agents.react_agent import ReactAgent, REACT_PROMPT
from .agents.simple_agent import SimpleAgent
from .agents.factory import create_agent

# --- 核心层 ---
from .core.agent import Agent, SubAgent
from .core.config import Config
from .core.exceptions import (
    AgentException,
    ConfigException,
    LLMException,
    ToolException,
    VioletAgentException,
)
from .core.llm import SUPPORTED_PROVIDERS, TOOL_CHOICE, VioletAgentsLLM
from .core.message import Message, MessageRole
from .core.session import Session

# --- 工具系统 ---
from .tools import (
    ApprovalTool,
    DefaultApprovalTool,
    SearchToolsTool,
    SkillsTool,
    TerminalTool,
    TodoWriteTool,
    Tool,
    ToolParameters,
    ToolProperty,
    ToolRegistry,
    WeatherTool,
)

__all__ = [
    # Agent 层
    "ReactAgent",
    "REACT_PROMPT",
    "SimpleAgent",
    "create_agent",
    # 核心层
    "Agent",
    "SubAgent",
    "Config",
    "Session",
    "VioletAgentsLLM",
    "SUPPORTED_PROVIDERS",
    "TOOL_CHOICE",
    "Message",
    "MessageRole",
    # 异常
    "AgentException",
    "ConfigException",
    "LLMException",
    "ToolException",
    "VioletAgentException",
    # 工具系统
    "Tool",
    "ToolParameters",
    "ToolProperty",
    "ToolRegistry",
    "ApprovalTool",
    "DefaultApprovalTool",
    # 内置工具
    "WeatherTool",
    "SkillsTool",
    "SearchToolsTool",
    "TerminalTool",
    "TodoWriteTool",
]
