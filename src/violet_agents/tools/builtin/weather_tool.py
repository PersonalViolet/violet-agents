
from typing import Dict, Any
from pydantic import BaseModel
from ..base import Tool, ToolParameters, ToolProperty
from ...core.message import Message, MessageRole


class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="获取指定城市的天气信息"
        )
    
    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        city = parameters.get("city")
        # 这里可以调用实际的天气API获取数据，以下是模拟数据
        weather_info = f"{city}的天气是晴朗，温度25°C。"
        return Message(
            content=weather_info,
            role="tool",
            tool_call_id=tool_call_id
        )
    
    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "city": ToolProperty(
                    type="string",
                    description="要查询天气的城市名称"
                )
            },
            required=["city"]
        )