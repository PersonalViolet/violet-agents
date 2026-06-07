
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core.message import Message
from pydantic import BaseModel


class ToolProperty(BaseModel):
    type: str
    description: str

class ToolParameters(BaseModel):
    """
    工具参数类，用于定义工具的输入参数结构，符合OpenAI API要求

    type: string - object
    properties: dict - 定义每个参数的类型和描述
    required: list[str] - 必须的参数列表
    """

    type: str
    properties: Dict[str, ToolProperty]
    required: list[str]

# class ToolParameter(BaseModel):
#     """
#     工具参数类，用于定义工具的输入参数
#     """
#     name: str
#     type: str
#     description: str
#     required: bool = True

class Tool(ABC):

    def __init__(self, 
                 name, 
                 description):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        """运行工具，处理输入参数并返回封装好的Message对象"""
        pass

    @abstractmethod
    def get_parameters(self) -> ToolParameters:
        """定义工具的输入参数结构，返回ToolParameters对象"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证输入参数是否包含必须的字段"""
        for param in self.get_parameters().required:
            if param not in parameters:
                return False
        return True


    def to_openai_dict(self) -> Dict[str, Any]:
        """转换为OpenAI API要求的工具格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters().model_dump()
            }
        }

    def get_session_state(self) -> Dict[str, Any]:
        """返回工具当前可变状态的 JSON 安全快照。有状态工具需重写此方法。"""
        return {}

    def restore_session_state(self, state: Dict[str, Any]) -> None:
        """从之前保存的状态快照恢复工具状态。有状态工具需重写此方法。"""
        pass

    def reset(self) -> None:
        """将工具重置为初始（__init__ 后）状态。有状态工具需重写此方法。"""
        pass

    