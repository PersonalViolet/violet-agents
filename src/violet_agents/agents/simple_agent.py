"""简单Agent，一次对话立即响应，不进行复杂的思考和计划"""

from typing import Optional
from ..core.config import Config
from ..core.agent import Agent
from ..core.llm import VioletAgentsLLM
from ..core.message import Message

class SimpleAgent(Agent):
    """简单Agent，一次对话立即响应，不进行复杂的思考和计划"""
    def __init__(self, 
                 name: str,
                 llm: VioletAgentsLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text) -> str:
        user_message = Message(content=input_text, role="user")
        self.add_message(user_message)
        messages = list(self.get_history())
        response = self.llm.chat(messages=messages)
        response_text = response.choices[0].message.content
        self.add_message(user_message)
        response_message = Message(content=response_text, role="assistant")
        self.add_message(response_message)
        return response_message