"""简单Agent，一次对话立即响应，不进行复杂的思考和计划"""

from typing import Optional
from ..core.config import Config
from ..core.agent import Agent
from ..core.llm import VioletAgentsLLM
from ..core.message import Message
from dotenv import load_dotenv

class SimpleAgent(Agent):
    """简单Agent，一次对话立即响应，不进行复杂的思考和计划"""
    def __init__(self, 
                 name: str,
                 llm: VioletAgentsLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text) -> Message:
        user_message = Message(content=input_text, role="user")
        history = self.get_history()
        history.append(user_message)
        messages = history
        response = self.llm.chat(messages=messages)
        response_text = response.choices[0].message.content
        self.add_message(user_message)
        response_message = Message(content=response_text, role="assistant")
        self.add_message(response_message)
        return response_message

if __name__ == "__main__":
    load_dotenv(dotenv_path="D:/My-Project/violet_agents/.env")
    llm = VioletAgentsLLM(provider='deepseek')
    agent = SimpleAgent(name="SimpleAgent", llm=llm, system_prompt="你是一个简单的助手，直接回答用户的问题，不进行复杂的思考和计划。")
    message = agent.run("简单介绍下agent")
    print(message.content)