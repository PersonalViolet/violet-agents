
from typing import Dict, Any
from pydantic import BaseModel
from ..base import Tool, ToolParameters, ToolProperty
from ...core.message import Message, MessageRole


class TodoWriteTool(Tool):
    def __init__(self):
        super().__init__(
            name="write_todo",
            description="写入待办事项"
        )
    
    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        todo = parameters.get("todo")
        # 这里可以调用实际的待办事项API获取数据，以下是模拟数据
        todo_info = f"已添加待办事项：{todo}"
    
    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "todo": ToolProperty(
                    type="string",
                    description="要添加的待办事项"
                )
            },
            required=["todo"]
        )