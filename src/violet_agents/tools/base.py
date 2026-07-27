
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..core.message import Message
from pydantic import BaseModel
import contextvars
import copy


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
        # 每个执行上下文（线程/asyncio task）持有独立的状态副本
        self._state_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = \
            contextvars.ContextVar(f"tool_state_{name}_{id(self)}", default=None)

    def _get_state(self) -> Dict[str, Any]:
        """获取当前执行上下文的工具可变状态。

        首次访问时懒初始化，避免 ContextVar 的 mutable-default 陷阱。
        子类通过此方法读写状态，天然线程/协程隔离。
        """
        state = self._state_var.get()
        if state is None:
            state = self._make_default_state()
            self._state_var.set(state)
        return state

    def _make_default_state(self) -> Dict[str, Any]:
        """返回工具的默认状态快照。有状态工具子类需重写此方法。"""
        return {}
    
    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        """运行工具，处理输入参数并返回封装好的Message对象"""
        raise NotImplementedError("子类需要实现 run() 方法")

    async def arun(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        """异步运行工具，处理输入参数并返回封装好的Message对象
        
        默认实现为同步调用run()，有异步需求的工具子类可重写此方法。"""
        return self.run(parameters, tool_call_id)
    
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
        """返回工具当前可变状态的深拷贝快照，用于跨 Session 持久化。

        从当前执行上下文的 ContextVar 读取状态，天然线程安全。
        有状态工具子类通常无需重写此方法。"""
        return copy.deepcopy(self._get_state())

    def restore_session_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """从之前保存的状态快照恢复到当前执行上下文。

        写入当前执行上下文的 ContextVar，其他线程/协程不受影响。
        传入 None 或空 dict 时自动回退到 _make_default_state()。
        子类仅需在需要对传入状态做验证/修正时重写，重写后必须调用
        super().restore_session_state(validated_state)。"""
        if state:
            self._state_var.set(copy.deepcopy(state))
        else:
            self._state_var.set(self._make_default_state())

    def reset(self) -> None:
        """将当前执行上下文的工具状态重置为默认值。

        有状态工具子类通常无需重写此方法。"""
        self._state_var.set(self._make_default_state())

    