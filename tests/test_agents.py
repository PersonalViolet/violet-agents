
import pytest
from src.violet_agents.agents.simple_agent import SimpleAgent
from src.violet_agents.core.llm import VioletAgentsLLM
from src.violet_agents.tools.registry import ToolRegistry
from src.violet_agents.agents.react_agent import ReactAgent
from src.violet_agents.tools.builtin.weather_tool import WeatherTool
from src.violet_agents.tools.base import Tool
from dotenv import load_dotenv

class TestAgent:
    def test_simple_agent_response(self):
        load_dotenv(dotenv_path="D:/My-Project/violet_agents/.env")
        llm = VioletAgentsLLM(provider='deepseek')
        agent = SimpleAgent(name="SimpleAgent", llm=llm, system_prompt="你是一个简单的助手，直接回答用户的问题，不进行复杂的思考和计划。")
        message = agent.run("简单介绍下agent")
        print("SimpleAgent response:", message.content)
        assert message.content is not None
        assert len(message.content) > 0

    def test_react_agent_response(self):
        load_dotenv(dotenv_path="D:/My-Project/violet_agents/.env")
        llm = VioletAgentsLLM(provider='deepseek')
        tool_registry = ToolRegistry()
        weather_tool = WeatherTool()
        tool_registry.register_tool(weather_tool)
        agent = ReactAgent(name="ReactAgent",
                           llm=llm,
                           max_steps=6,
                           tool_registry=tool_registry)
        message = agent.run("北京天气怎么样？上海天气如何？南京呢？东京呢？纽约呢？广州呢？深圳呢？")
        assert message.content is not None
        assert len(message.content) > 0


        