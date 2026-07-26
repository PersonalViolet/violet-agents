"""MCP (Model Context Protocol) 客户端 —— 基于 fastmcp 实现。

当前处于早期开发阶段，提供基础的 MCP 客户端连接能力。
"""

from .client import FASTMCP_AVAILABLE
from .client import MCPClient
__all__ = [
    "FASTMCP_AVAILABLE", 
    "MCPClient"]
