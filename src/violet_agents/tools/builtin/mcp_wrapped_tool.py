from typing import Any, Dict, TYPE_CHECKING
from ..base import Tool, ToolParameters, ToolProperty
from ...core.message import Message
if TYPE_CHECKING:
    from . import MCPTool

class MCPWrappedTool(Tool):
    """
    MCP工具包装器 - 将单个MCP工具包装成HelloAgents Tool
    
    这个类将MCP服务器的一个工具（如 read_file）包装成一个独立的Tool对象。
    Agent调用时只需提供参数，无需了解MCP的内部结构。
    
    示例：
        >>> # 内部使用，由MCPTool自动创建
        >>> wrapped_tool = MCPWrappedTool(
        ...     mcp_tool=mcp_tool_instance,
        ...     tool_info={
        ...         "name": "read_file",
        ...         "description": "Read a file...",
        ...         "input_schema": {...}
        ...     }
        ... )
    """

    def __init__(self, 
                 mcp_tool: 'MCPTool',
                 tool_info: Dict[str, Any],
                 prefix: str = ""):
        """
        初始化MCP包装工具

        Args:
            mcp_tool: 父MCP工具实例
            tool_info: MCP工具信息（包含name, description, input_schema）
            prefix: 工具名前缀（如 "filesystem_"）
        """
        self.mcp_tool = mcp_tool
        self.tool_info = tool_info
        self.prefix = prefix

        self.mcp_tool_name = tool_info.get("name", "unknown")
        # 构建工具名： prefix + mcp_tool_name
        tool_name = f"{self.prefix}{self.mcp_tool_name}" if prefix else self.mcp_tool_name
        # 获取描述
        description = tool_info.get("description", f'MCP工具: {self.mcp_tool_name}')
        # 解析参数schema
        self._parameters = self._parse_input_schema(tool_info.get('input_schema', {}))

        super().__init__(name=tool_name, description=description)

    def _parse_input_schema(self, input_schema: Dict[str, Any]) -> ToolParameters:
        """
        将MCP的input_schema转换为ToolParameters

        MCP的input_schema是JSON Schema格式，例如：
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "mode": {"type": "string", "description": "打开模式"}
                },
                "required": ["path"]
            }

        Args:
            input_schema: MCP工具的input_schema（JSON Schema格式）

        Returns:
            ToolParameters
        """
        properties: Dict[str, ToolProperty] = {}
        for prop_name, prop_schema in input_schema.get("properties", {}).items():
            properties[prop_name] = ToolProperty(
                type=prop_schema.get("type", "string"),
                description=prop_schema.get("description", "")
            )

        return ToolParameters(
            type=input_schema.get("type", "object"),
            properties=properties,
            required=input_schema.get("required", [])
        )


    def get_parameters(self) -> ToolParameters:
        return self._parameters

    def run(self, params: Dict[str, Any], tool_call_id: str) -> Message:
        """
        执行MCP工具

        Args:
            params: 工具参数（直接传递给MCP工具）
            tool_call_id: 工具调用ID

        Returns:
            执行结果
        """
        mcp_params = {
            "action": "call_tool",
            "tool_name": self.mcp_tool_name,
            "arguments": params
        }
        # 调用父MCP工具的run方法
        return self.mcp_tool.run(mcp_params, tool_call_id=tool_call_id)



        