from typing import Optional, List, Any, Dict, Callable, Literal, overload, TYPE_CHECKING, Union
from ..core.config import Config
from ..core.agent import Agent
from ..core.llm import VioletAgentsLLM
from ..core.message import Message
from ..core.session import Session
from ..tools.registry import ToolRegistry
from dotenv import load_dotenv
from ..tools.builtin.weather_tool import WeatherTool
from ..tools.builtin.terminal_tool import TerminalTool
from ..tools.builtin.skills_tool import SkillsTool
from ..tools.builtin.search_tools_tool import SearchToolsTool
from ..tools.builtin.mcp_tool import MCPTool
from ..tools import ConsoleConfirmInterceptor
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
from collections import deque
import json
import os

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
    """ReactAgent，基于ReAct框架设计的Agent，支持会话隔离。

    不变配置：llm、tool_registry、max_steps、system_prompt。
    可变状态全部存储在 Session 中，每个 session 拥有独立的：
    消息历史、临时工具、钩子、轮次计数、工具状态快照。
    """


    def __init__(self,
                 name: str,
                 llm: VioletAgentsLLM,
                 max_steps: int = 5,
                 tool_registry: Optional[ToolRegistry] = None,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        super().__init__(name, llm, system_prompt, config, tool_registry)
        self.max_steps = max_steps

        self.register_session_hook("PostToolCall", self._handle_search_tools_hook)
        self.register_session_hook("PreToolCall", self._on_temp_tool_called_hook)



    # --- 主运行循环 ---
    def do_run(self, 
               input_text: str, 
               session: "Session", 
               **kwargs) -> Message:
        sess = session
        sess.agent_state.setdefault("current_round", 0)
        self._trigger_session_hooks("UserPromptSubmit", input_text, sess=sess)

        user_message = Message(content=input_text, role="user")
        sys_message = Message(content=self.system_prompt, role="system") if self.system_prompt else None

        messages = list(sess.get_history())  # 转为 list，后续追加
        if sys_message:
            messages.append(sys_message)
        messages.append(user_message)

        current_step = 0
        while current_step < self.max_steps:
            tools = self.tool_registry.get_openai_tools() + sess.agent_state.get("temp_tools", [])
            response = self.llm.chat(messages=messages, tools=tools, tool_choice="auto")
            message = response.choices[0].message
            tool_calls = message.tool_calls

            if not tool_calls:
                response_message = Message.from_chat_completion_message(message)
                messages.append(response_message)
                # 修复 bug：用 deque 保持类型一致并应用 maxlen
                sess._history = deque(messages, maxlen=sess.max_history_length if sess.max_history_length > 0 else None)
                sess._touch()
                return response_message

            llm_message = Message.from_chat_completion_message(message)
            messages.append(llm_message)
            # 使用并发执行，确保 PreToolCall / PostToolCall 钩子对每条 tool_call 都生效
            tool_response_messages = self.execute_tools_concurrently(tool_calls)
            messages.extend(tool_response_messages)
            current_step += 1

        response_text = "超出最大思考步骤限制，无法给出回答。"
        response_message = Message(content=response_text, role="assistant")
        messages.append(response_message)
        sess._history = deque(messages, maxlen=sess.max_history_length if sess.max_history_length > 0 else None)
        sess._touch()
        # 更新agent_state中的current_round
        sess.agent_state["current_round"] += 1
        return response_message

    # --- 主运行循环 ---
    async def ado_run(self, 
               input_text: str, 
               session: "Session", 
               **kwargs) -> Message:
        sess = session
        sess.agent_state.setdefault("current_round", 0)
        self._trigger_session_hooks("UserPromptSubmit", input_text, sess=sess)

        user_message = Message(content=input_text, role="user")
        sys_message = Message(content=self.system_prompt, role="system") if self.system_prompt else None

        messages = list(sess.get_history())  # 转为 list，后续追加
        if sys_message:
            messages.append(sys_message)
        messages.append(user_message)

        current_step = 0
        while current_step < self.max_steps:
            tools = self.tool_registry.get_openai_tools() + sess.agent_state.get("temp_tools", [])
            response = await self.llm.achat(messages=messages, tools=tools, tool_choice="auto")
            message = response.choices[0].message
            tool_calls = message.tool_calls

            if not tool_calls:
                response_message = Message.from_chat_completion_message(message)
                messages.append(response_message)
                sess._history = deque(messages, maxlen=sess.max_history_length if sess.max_history_length > 0 else None)
                sess._touch()
                return response_message

            llm_message = Message.from_chat_completion_message(message)
            messages.append(llm_message)
            tool_response_messages = await self.aexecute_tools(tool_calls)
            messages.extend(tool_response_messages)
            current_step += 1

        response_text = "超出最大思考步骤限制，无法给出回答。"
        response_message = Message(content=response_text, role="assistant")
        messages.append(response_message)
        sess._history = deque(messages, maxlen=sess.max_history_length if sess.max_history_length > 0 else None)
        sess._touch()
        # 更新agent_state中的current_round
        sess.agent_state["current_round"] += 1
        return response_message

    def execute_tool(self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]) -> Message:
        if self.tool_registry is None:
            raise RuntimeError("ToolRegistry is not initialized.")
        sess = self._get_active_session()
        self._trigger_session_hooks("PreToolCall", tool_call, sess=sess)
        tool_response_message = self.tool_registry.execute_tool(tool_call)
        self._trigger_session_hooks("PostToolCall", tool_response_message, sess=sess)
        return tool_response_message

    async def aexecute_tool(self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]) -> Message:
        """异步执行工具调用。
        """
        if self.tool_registry is None:
            raise RuntimeError("ToolRegistry is not initialized.")
        sess = self._get_active_session()
        self._trigger_session_hooks("PreToolCall", tool_call, sess=sess)
        tool_response_message = await self.tool_registry.aexecute_tool(tool_call)
        self._trigger_session_hooks("PostToolCall", tool_response_message, sess=sess)
        return tool_response_message
    # --- 临时工具管理（操作 Session 中的字段） ---

    def _add_temp_tool(self, tool_schema_dict: Dict[str, Any], sess: Optional[Session] = None) -> None:
        s = sess or self._get_active_session()
        if s is None:
            return
        s.agent_state.setdefault("temp_tools", []).append(tool_schema_dict)
        s.agent_state.setdefault("temp_tools_names", set()).add(tool_schema_dict.get("function", {}).get("name"))

    def _cleanup_temp_tools(self, tools_to_remove: List[str], sess: Optional[Session] = None) -> None:
        s = sess or self._get_active_session()
        if s is None:
            return
        s.agent_state["temp_tools"] = [tool for tool in s.agent_state.get("temp_tools", [])
                        if tool.get("function", {}).get("name") not in tools_to_remove]
        for tool_name in tools_to_remove:
            s.agent_state["temp_tools_names"].discard(tool_name)
            if tool_name in s.agent_state.get("temp_tools_last_call_round", {}):
                del s.agent_state["temp_tools_last_call_round"][tool_name]

    def _check_temp_tools_expiry(self, sess: Optional[Session] = None) -> List[str]:
        s = sess or self._get_active_session()
        if s is None:
            return []
        expired_tools = []
        for tool_name, last_call_round in s.agent_state.get("temp_tools_last_call_round", {}).items():
            if s.agent_state["current_round"] - last_call_round >= 3:
                expired_tools.append(tool_name)
        return expired_tools

    # --- 内置钩子回调 ---

    def _handle_search_tools_hook(self, tool_response_message: Message) -> None:
        sess = self._get_active_session()
        if tool_response_message.tool_call_id and tool_response_message.metadata \
                and tool_response_message.metadata.get("tool_type") == SearchToolsTool \
                and tool_response_message.metadata.get("action") == "get":
            tool_schema = tool_response_message.content
            try:
                tool_schema_dict = json.loads(tool_schema)
                self._add_temp_tool(tool_schema_dict, sess=sess)
                tool_response_message.content = f"已加载工具: {tool_schema_dict.get('function', {}).get('name', '未知工具')}"
            except json.JSONDecodeError as e:
                print(f"工具schema解析失败，确保工具返回的schema是有效的JSON字符串: {e}")

    def _on_temp_tool_called_hook(self, tool_call: ChatCompletionMessageFunctionToolCall) -> None:
        sess = self._get_active_session()
        if sess is None:
            return
        # 兼容 dict 和 ChatCompletionMessageFunctionToolCall 两种格式
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("function", {}).get("name", "")
        else:
            tool_name = tool_call.function.name
        if tool_name in sess.agent_state.get("temp_tools_names", set()):
            sess.agent_state.setdefault("temp_tools_last_call_round", {})[tool_name] = sess.agent_state["current_round"]
            expired_tools = self._check_temp_tools_expiry(sess=sess)
            self._cleanup_temp_tools(expired_tools, sess=sess)

    if TYPE_CHECKING:
        @overload
        def register_session_hook(self, event: Literal["UserPromptSubmit"],
                        callback: Callable[[str], None],
                        session_id: Optional[str] = None) -> None: ...

        @overload
        def register_session_hook(self, event: Literal["PreToolCall"],
                        callback: Callable[[ChatCompletionMessageFunctionToolCall], None],
                        session_id: Optional[str] = None) -> None: ...

        @overload
        def register_session_hook(self, event: Literal["PostToolCall"],
                        callback: Callable[[Message], None],
                        session_id: Optional[str] = None) -> None: ...
        
        @overload
        def _trigger_session_hooks(self, event: Literal["UserPromptSubmit"], arg: str, sess: Optional["Session"] = None) -> None: ...

        @overload
        def _trigger_session_hooks(self, event: Literal["PreToolCall"], arg: ChatCompletionMessageFunctionToolCall, sess: Optional["Session"] = None) -> None: ...

        @overload
        def _trigger_session_hooks(self, event: Literal["PostToolCall"], arg: Message, sess: Optional["Session"] = None) -> None: ...


if __name__ == "__main__":

    tool_registry = ToolRegistry(ConsoleConfirmInterceptor(intercept_list=[TerminalTool],
                                                     max_attempts=5,
                                                     auto_approve_if_no_rules=True))
    tool_registry.register_tools(WeatherTool(), 
                                 SkillsTool(), 
                                 SearchToolsTool(get_deferTools_callback=tool_registry.get_defer_tools, search_strategy="subAgent"), 
                                 is_defer=False)
    tool_registry.register_tool(TerminalTool(), is_defer=False)
    agent = ReactAgent(
        name="MyAgent",
        llm=VioletAgentsLLM(provider="deepseek"),
        tool_registry=tool_registry,
        max_steps=10,
    )

    agent.registry_agent_hook("SessionInit", lambda sess: print(f"Session '{sess.session_id}' 正在初始化"))
    agent.registry_agent_hook("PreSessionSwitch", lambda sessions: print(f"正在切换 session，从 '{sessions[0].session_id if sessions[0] else None}' 切换到 '{sessions[1].session_id}'"))
    # 方式一：指定 session_id 运行（自动创建 session）
    agent.create_session("user-123")
    agent.register_session_hook("PostToolCall", lambda msg: print(f"工具调用后返回了: {msg.content}"), session_id="user-123")
    agent.register_session_hook("PreToolCall", lambda tool_call: print(f"即将调用工具: {tool_call.function.name}\n"), session_id="user-123")
    config = {
        "mcpServers": {
            # --- HTTP/Streamable HTTP 远程服务器 ---
            "github": {
                "url": "https://api.githubcopilot.com/mcp/",
                "transport": "http",               # "http" | "streamable-http" | "sse"
                "headers": {                      # 自定义 HTTP 请求头
                    "Authorization": f"Bearer {os.getenv('GITHUB_COPILOT_TOKEN')}",
                },
                # "auth": "oauth",                  # 认证方式: str (Bearer token) | "oauth" | httpx.Auth
                "timeout": 30000,                   # 超时时间 (毫秒)
            },
            "tavily": {
                "url": "https://mcp.tavily.com/mcp",
                "transport": "http",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('TAVILY_MCP_TOKEN')}",
                },
            },
            # --- Stdio 本地进程服务器（示例） ---
            # "fetch": {
            #     "command": "uvx",                 # 必填: npx | uvx | python | node | ...
            #     "args": ["mcp-server-fetch"],     # 命令行参数
            #     # "env": {"API_KEY": "xxx"},      # 环境变量
            #     # "cwd": "/path/to/workdir",      # 工作目录
            #     "transport": "stdio",             # 默认 stdio，可省略
            #     "timeout": 30000,                 # 超时时间 (毫秒)
            #     "description": "网页抓取 MCP 服务器",
            # },
            # --- 从 Python 脚本启动 ---
            # "my-tools": {
            #     "command": "python",
            #     "args": ["-m", "my_mcp_server"],
            #     "transport": "stdio",
            #     "description": "自定义 MCP 工具集",
            # },
        }
    }
    MCPTool(server_source=config, auto_expand=False).register_to(tool_registry)
    print("异步运行开始")
    async def run_agent_tasks():
        response = await agent.arun("你好，我github的账户名称叫什么（使用github mcp服务查询）？看看当前项目文件夹有什么？并行执行这些操作", session_id="user-123")
        print(response.content)
        response = await agent.arun("帮我看下这些域名的IP地址：baidu.com, github.com, deepseek.com", session_id="user-123")
        print(response.content)
    import asyncio
    asyncio.run(run_agent_tasks())
    print("异步运行结束")
    response = agent.run("你好，我github的账户名称叫什么？我的github有什么仓库？https://www.bilibili.com/video/BV1BvgQ6iEn9?spm_id_from=333.1007.tianma.1-1-1.click这个网址简要介绍一下内容，并行执行这些操作。意思是同时执行三个mcp_tool提供的call_tool", session_id="user-123")
    print(response.content)
    response = agent.run("帮我看下这些域名的IP地址：baidu.com, github.com, deepseek.com", session_id="user-123")
    print(response.content)
    agent.deactivate_session()  # 显式结束 session，触发工具状态保存

    # 方式二：上下文管理器
    with agent.session("user-123"):
        response = agent.run("你肯定有terminal工具，想方设法找到它。简单看看我的ip地址，还有我刚刚询问天气的城市叫啥？")
        print(response.content)
        response = agent.run("使用cd命令（不要用/d）进入我的src/violet_agents/agents包；看我的agents包里有什么文件？")
        print(response.content)

    # 方式三：手动管理
    agent.create_session("user-456")
    agent.switch_session("user-456")
    response = agent.run("我刚刚说了什么？还有，我喜欢千小妹，简短回复我")
    print(response.content)
    response = agent.run("terminal工具现在的工作目录在哪里？看一下并告诉我，使用cd命令检查一下，你一定有terminal工具的，你用search_tools工具找一下，如果找不到就直接调用看看，不要放弃！")
    print(response.content)
    agent.switch_session("user-123")  # 切回之前的 session
    response = agent.run("我之前说了什么？现在的terminal工具工作目录在哪里？使用cd命令检查")
    print(response.content)