from typing import Optional, List, Any, Dict, Callable, Literal, overload
from ..core.config import Config
from ..core.agent import Agent
from ..core.llm import VioletAgentsLLM
from ..core.message import Message
from ..tools.registry import ToolRegistry
from dotenv import load_dotenv
from ..tools.builtin.weather_tool import WeatherTool
from ..tools.builtin.terminal_tool import TerminalTool
from ..tools.builtin.skills_tool import SkillsTool
from ..tools.builtin.search_tools_tool import SearchToolsTool
from ..tools import DefaultApprovalTool
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
import json

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


HookEvent = Literal["UserPromptSubmit", "PreToolCall", "PostToolCall"]
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
        self.temp_tools: List[Dict[str, Any]] = []  # 临时工具列表，用于存储当前对话中发现的工具信息
        self.temp_tools_names: set = set()  # 临时工具名称集合，辅助快速判断工具是否已存在
        self.temp_tools_last_call_round: Dict[str, int] = {} # 记录temp_tools中每个工具上次被调用的轮次，用于实现工具过期机制，key为工具名称，value为上次被调用的轮次
        self.current_round: int = 0 # 当前对话轮次计数器
        self.hooks: Dict[HookEvent, List[Callable]] = {"UserPromptSubmit": [], "PreToolCall": [], "PostToolCall": []}
        self.register_hooks("PostToolCall", self._handle_search_tools_hook)
        self.register_hooks("PreToolCall", self._on_temp_tool_called_hook)

    @overload
    def register_hooks(self, event: Literal["UserPromptSubmit"], callback: Callable[[str], None]) -> None: 
        """
            注册UserPromptSubmit事件的钩子
            
            :param event: "UserPromptSubmit" 事件
            :param callback: 回调函数，接收用户输入文本作为参数

        """
        ...

    @overload
    def register_hooks(self, event: Literal["PreToolCall"], callback: Callable[[ChatCompletionMessageFunctionToolCall], None]) -> None: 
        """
            注册PreToolCall事件的钩子
            
            :param event: "PreToolCall" 事件
            :param callback: 回调函数，接收ChatCompletionMessageFunctionToolCall类型作为参数
        """
        ...
    @overload
    def register_hooks(self, event: Literal["PostToolCall"], callback: Callable[[Message], None]) -> None: 
        """
            注册工具调用后事件的钩子
            
            :param event: "PostToolCall" 事件
            :param callback: 回调函数，接收工具调用结果相关参数
        """
        ...


    def register_hooks(self, event: HookEvent, callback: Callable):
        """
            注册用户输入提交事件的钩子
            
            :param event: "UserPromptSubmit" 事件
            :param callback: 回调函数，接收用户输入文本作为参数
        """
        if event in self.hooks:
            self.hooks[event].append(callback)

    def _trigger_hooks(self, event: HookEvent, *args):
        """
        触发钩子函数
        Args:
            event (HookEvent): 事件类型
            *args: 根据事件类型不同而不同的参数列表
        """
        for callback in self.hooks[event]:
            result = callback(*args)
            if result is not None:  # 返回值 ≠ None → hook 说"停"
                return result
            return None

    def run(self, input_text) -> Message:
        user_message = Message(content=input_text, role="user")
        sys_message = Message(content=self.system_prompt, role="system") if self.system_prompt else None
        messages = self.get_history()
        if sys_message:
            messages.append(sys_message)
        messages.append(user_message)
        current_step = 0
        while current_step < self.max_steps:
            tools = self.tool_registry.get_openai_tools() + self.temp_tools
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
                self._trigger_hooks("PreToolCall", tool_call)
                tool_response_message = self.tool_registry.execute_tool(tool_call)
                self._trigger_hooks("PostToolCall", tool_response_message)
                messages.append(tool_response_message)
            current_step += 1
        # 超过最大步骤限制，返回最后一次回答
        response_text = "超出最大思考步骤限制，无法给出回答。"
        response_message = Message(content=response_text, role="assistant")
        messages.append(response_message)
        self._history = messages
        self.current_round += 1
        return response_message
    
    def _add_temp_tool(self, tool_schema_dict: Dict[str, Any]) -> None:
        """添加临时工具
        将新的工具信息添加到temp_tools列表和temp_tools_names集合中

        Args:
            tool_schema_dict (Dict[str, Any]): 工具信息字典，包含工具的名称、功能描述、参数等信息
        """
        self.temp_tools.append(tool_schema_dict)
        self.temp_tools_names.add(tool_schema_dict.get("function", {}).get("name"))

    def _cleanup_temp_tools(self, tools_to_remove: List[str]) -> None:
        """删除过时的临时工具
        从temp_tools, temp_tools_last_call_round中删除过时工具，并更新temp_tools_names集合
        
        Args:
            tools_to_remove (List[str]): 需要删除的工具名称列表
        """
        self.temp_tools = [tool for tool in self.temp_tools 
                            if tool.get("function", {}).get("name") not in tools_to_remove]
        for tool_name in tools_to_remove:
            self.temp_tools_names.discard(tool_name)
            del self.temp_tools_last_call_round[tool_name]

    def _check_temp_tools_expiry(self) -> List[str]:
        """检查临时工具过期
        检查temp_tools_last_call_round中记录的工具调用轮次，如果当前轮次与上次调用轮次之差超过3轮，则认为该工具过期，返回过期工具名称列表

        Returns:
            List[str]: 过期工具名称列表
        """
        current_round = self.current_round
        expired_tools = []
        for tool_name, last_call_round in self.temp_tools_last_call_round.items():
            if current_round - last_call_round >= 3:
                expired_tools.append(tool_name)
        return expired_tools



    def _handle_search_tools_hook(self, tool_response_message: Message) -> None:
        """处理搜索工具返回内容的PostToolCall钩子函数
        当工具调用返回后，如果是搜索工具的get功能的调用结果，则解析工具返回的内容，提取工具信息并将其添加到临时工具列表中，以便后续对话中可以使用这些新工具
        
        Args:
            tool_response_message (Message): 工具调用返回的消息对象，包含工具调用结果的相关信息
        """
        if tool_response_message.tool_call_id and tool_response_message.metadata and tool_response_message.metadata.get("tool_type") == SearchToolsTool and tool_response_message.metadata.get("action") == "get":
            tool_schema = tool_response_message.content
            try:
                tool_schema_dict = json.loads(tool_schema)
                self._add_temp_tool(tool_schema_dict)
                tool_response_message.content = f"已加载工具: {tool_schema_dict.get('function', {}).get('name', '未知工具')}"
            except json.JSONDecodeError as e:
                print(f"工具schema解析失败，确保工具返回的schema是有效的JSON字符串: {e}")

    def _on_temp_tool_called_hook(self, tool_call: ChatCompletionMessageFunctionToolCall) -> None:
        """这是一个PreToolCall钩子函数，用于处理临时工具的过期检查和调用轮次更新
        当工具调用发生时，如果调用的工具在temp_tools中，则更新该工具的最后调用轮次为当前轮次，并检查temp_tools中是否有工具超过三轮未被调用，如果有则将其从temp_tools中删除

        Args:
            tool_call (ChatCompletionMessageFunctionToolCall): 当前的工具调用信息，包含工具名称、参数等信息
        """
        tool_name = tool_call.function.name
        # 只有当调用的工具在temp_tools中时才更新调用轮次并检查过期工具
        if tool_name in self.temp_tools_names:
            self.temp_tools_last_call_round[tool_name] = self.current_round
            # 检查过期工具
            expired_tools = self._check_temp_tools_expiry()
            
            # 从temp_tools和temp_tools_last_call_round中删除这些工具
            self._cleanup_temp_tools(expired_tools)


if __name__ == "__main__":
    load_dotenv(dotenv_path="D:/My-Project/violet_agents/.env")
    llm = VioletAgentsLLM(provider='deepseek')
    tool_registry = ToolRegistry(DefaultApprovalTool(require_approval_tools=[TerminalTool],
                                                     max_attempts=5,
                                                     auto_approve_if_no_rules=True))
    weather_tool = WeatherTool()
    terminal_tool = TerminalTool()
    skills_tool = SkillsTool()
    search_tools_tool = SearchToolsTool(get_deferTools_callback=tool_registry.get_defer_tools, search_strategy="subAgent")
    tool_registry.register_tool(weather_tool)
    tool_registry.register_tool(terminal_tool, is_defer=True)  # 将终端工具注册为延迟工具
    tool_registry.register_tool(skills_tool)
    tool_registry.register_tool(search_tools_tool)
    system_msg = REACT_PROMPT + "\n\n" + skills_tool.get_system_prompt_section()
    agent = ReactAgent(name="ReactAgent", llm=llm, system_prompt=system_msg, tool_registry=tool_registry, max_steps=10)
    # message = agent.run("看一下今天北京的天气，并看看我的端口占用情况，还有我目前连接的网络里还有哪些设备？")
    message = agent.run("你好啊，我想看看我目前连接的网络里还有哪些设备？使用search_tools工具")
    print(message.content)
    
