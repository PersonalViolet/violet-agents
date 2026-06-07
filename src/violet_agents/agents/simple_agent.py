"""简单Agent，一次对话立即响应，不进行复杂的思考和计划"""

from typing import Optional
from ..core.config import Config
from ..core.agent import Agent
from ..core.llm import VioletAgentsLLM
from ..core.message import Message
from dotenv import load_dotenv
from ..core.session import Session


class SimpleAgent(Agent):
    """简单Agent，一次对话立即响应，不进行复杂的思考和计划"""

    def __init__(self,
                 name: str,
                 llm: VioletAgentsLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        super().__init__(name, llm, system_prompt, config)

    def do_run(self, 
               input_text: str, 
               session: "Session") -> Message:
        sess = session

        user_message = Message(content=input_text, role="user")
        history = sess.get_history()
        history.append(user_message)
        messages = history

        response = self.llm.chat(messages=messages)
        response_text = response.choices[0].message.content
        response_message = Message(content=response_text, role="assistant")

        sess.add_message(user_message)
        sess.add_message(response_message)
        # self._deactivate_session()
        return response_message


# if __name__ == "__main__":
#     load_dotenv(dotenv_path="D:/My-Project/violet_agents/.env")
#     llm = VioletAgentsLLM(provider='deepseek')
#     agent = SimpleAgent(name="SimpleAgent", llm=llm, system_prompt="你是一个简单的助手，直接回答用户的问题，不进行复杂的思考和计划。")
#     message = agent.run("简单介绍下agent")
#     print(message.content)

if __name__ == "__main__":
    agent = SimpleAgent(
        name="MyAgent",
        llm=VioletAgentsLLM(provider="deepseek"),
        system_prompt="你是一个简单的助手，直接回答用户的问题，不进行复杂的思考和计划。"
    )

    # 方式一：指定 session_id 运行（自动创建 session）
    response = agent.run("你好，达尼亚好可爱，简短回复我", session_id="user-123")
    print(response.content)

    # 方式二：上下文管理器
    with agent.session("user-123"):
        response = agent.run("我刚刚说了什么？")
    print(response.content)

    # 方式三：手动管理
    agent.create_session("user-456")
    agent.switch_session("user-456")
    response = agent.run("我刚刚说了什么？还有，我喜欢千小妹，简短回复我")
    print(response.content)
    agent.switch_session("user-123")  # 切回之前的 session
    response = agent.run("我有说我喜欢千小妹吗？")
    print(response.content)