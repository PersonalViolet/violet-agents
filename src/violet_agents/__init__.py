"""Public package API for violet_agents."""

from .agents.react_agent import ReactAgent, REACT_PROMPT
from .agents.simple_agent import SimpleAgent
from .core.agent import Agent
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
from .tools.base import Tool, ToolParameters, ToolProperty
from .tools.registry import ToolRegistry
from .tools.builtin.weather_tool import WeatherTool

__all__ = [
    "Agent",
    "AgentException",
    "Config",
    "ConfigException",
    "LLMException",
    "Message",
    "MessageRole",
    "ReactAgent",
    "REACT_PROMPT",
    "Session",
    "SimpleAgent",
    "SUPPORTED_PROVIDERS",
    "TOOL_CHOICE",
    "Tool",
    "ToolException",
    "ToolParameters",
    "ToolProperty",
    "ToolRegistry",
    "VioletAgentException",
    "VioletAgentsLLM",
    "WeatherTool",
]
