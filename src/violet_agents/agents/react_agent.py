from typing import Optional
from ..core.config import Config
from ..core.agent import Agent
from ..core.llm import VioletAgentsLLM
from ..core.message import Message
from ..tools.registry import ToolRegistry
from dotenv import load_dotenv
from ..tools.builtin.weather_tool import WeatherTool


REACT_PROMPT = """你是一个有能力使用工具的 AI 助手。
遇到需要计算、查询或获取信息的问题时，请主动调用相关工具来获取准确结果。

工作流程：
1. 分析用户问题，判断是否需要调用工具
2. 如需工具，调用对应的函数获取数据
3. 根据工具返回结果进行推理
4. 最终用中文给出清晰完整的回答

注意：
- 可以在一轮中同时调用多个独立的工具
- 每次工具调用后，仔细分析返回结果再决定下一步"""

class ReactAgent(Agent):
    """ReactAgent，基于ReAct框架设计的Agent，能够在对话过程中主动调用工具进行思考和行动
    """
    def __init__(self, 
                 name: str,
                 llm: VioletAgentsLLM,
                 max_steps: int = 5,
                 tool_registry: Optional[ToolRegistry] = None,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        super().__init__(name, llm, system_prompt, config)
        self.max_steps = max_steps
        self.tool_registry = tool_registry or ToolRegistry()

    def run(self, input_text) -> Message:
        user_message = Message(content=input_text, role="user")
        messages = self.get_history()
        messages.append(user_message)
        current_step = 0
        while current_step < self.max_steps:
            tools = self.tool_registry.get_openai_tools()
            response = self.llm.chat(messages=messages, tools=tools, tool_choice="auto")
            message = response.choices[0].message
            tool_calls = message.tool_calls
            if not tool_calls:
                # 没有工具调用，直接返回回答
                response_message = Message.from_chat_completion_message(message)
                messages.append(response_message)
                self._history = messages
                return response_message
            # 处理工具调用
            response_text = message.content if message.content else ""
            # llm_message = Message(content=response_text, role="assistant", tool_calls=tool_calls)
            llm_message = Message.from_chat_completion_message(message)
            messages.append(llm_message)
            for tool_call in tool_calls:
                tool_response_message = self.tool_registry.execute_tool(tool_call)
                messages.append(tool_response_message)
            current_step += 1
        # 超过最大步骤限制，返回最后一次回答
        response_text = "超出最大思考步骤限制，无法给出回答。"
        response_message = Message(content=response_text, role="assistant")
        messages.append(response_message)
        self._history = messages
        return response_message
    
if __name__ == "__main__":
    load_dotenv(dotenv_path="D:/My-Project/violet_agents/.env")
    llm = VioletAgentsLLM(provider='deepseek')
    tool_registry = ToolRegistry()
    weather_tool = WeatherTool()
    tool_registry.register_tool(weather_tool)
    agent = ReactAgent(name="ReactAgent", llm=llm, system_prompt=REACT_PROMPT, tool_registry=tool_registry)
    message = agent.run("北京天气怎么样？上海天气如何？南京呢？东京呢？纽约呢？广州呢？深圳呢？")
    print(message.content)
    
