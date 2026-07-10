"""工具系统 —— 工具基类、注册表、审批机制与内置工具。"""

# 基础抽象
from .base import Tool, ToolProperty, ToolParameters

# 注册表
from .registry import ToolRegistry

# 审批
from .approval_tool import ApprovalTool
from .infrastructure.default_approval_tool import DefaultApprovalTool

# 内置工具（从 builtin 子包导入）
from .builtin import (
    WeatherTool,
    SkillsTool,
    SearchToolsTool,
    TerminalTool,
    TodoWriteTool,
)

__all__ = [
    # 基础抽象
    "Tool",
    "ToolProperty",
    "ToolParameters",
    # 注册表
    "ToolRegistry",
    # 审批
    "ApprovalTool",
    "DefaultApprovalTool",
    # 内置工具
    "WeatherTool",
    "SkillsTool",
    "SearchToolsTool",
    "TerminalTool",
    "TodoWriteTool",
]