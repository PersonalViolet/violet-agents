
from typing import Dict, Any, Callable, Literal, Optional, Deque
from pydantic import BaseModel
from ..base import Tool, ToolParameters, ToolProperty
from ...core.message import Message, MessageRole
from ..registry import ToolRegistry
from ...core.agent import Agent, SubAgent
from ...core.llm import VioletAgentsLLM
from ...core.config import Config
import json
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...core.session import Session
SearchStrategy = Literal["keyword", "subAgent"]

logger = logging.getLogger(__name__)



class SearchToolsSubAgent(SubAgent):
    """简单Agent，一次对话立即响应，不进行复杂的思考和计划"""
    def __init__(self, 
                 name: str,
                 llm: Optional[VioletAgentsLLM] = None,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        super().__init__(name, llm, system_prompt, config)

    def do_run(self, 
               input_text,
               session: "Session",
               **kwargs) -> Message:
        sys_message = Message(content=self.system_prompt, role="system") if self.system_prompt else None
        user_message = Message(content=input_text, role="user")
        history = Deque()
        if sys_message:
            history.append(sys_message)
        history.append(user_message)
        messages = history
        response = self.llm.chat(messages=messages)
        response_message = Message.from_chat_completion(response)
        return response_message



class SearchToolsTool(Tool):
    """一个专门用于搜索工具的工具，支持基于子Agent的搜索和基于向量化的搜索两种策略
    action参数用于指定当前的操作类型，支持两种操作：search表示搜索相关工具，get表示获取具体工具信息。两者的Message.content格式都为JSON字符串
    当Agent使用该工具的search功能时，可以传入一个查询字符串，工具会根据查询字符串在当前的工具注册表中搜索相关工具，并返回相关工具的列表供Agent选择。
    Agent使用该工具的get功能后，工具会返回该工具的完整信息（名称、描述、参数等），供Agent调用执行。

    Attr:
        get_deferTools_callback (Callable[[], Dict[str, Tool]]): 一个回调函数，用于获取当前的懒加载工具列表，返回值是一个字典，键为工具名称，值为Tool对象
        search_strategy (SearchStrategy): 搜索策略，支持"keyword"和"subAgent"两种策略，默认为"keyword"
    """

    def __init__(self,
                 get_deferTools_callback: Callable[[], Dict[str, Tool]],
                 search_strategy: SearchStrategy):
        super().__init__(
            name="search_tools",
            description="""搜索工具的工具。
            当Agent在对话中判断需要调用工具，但现有的tools列表中没有合适的工具时，可以调用这个工具来搜索相关工具；
            当Agent需要获取某个工具的详细信息如参数时，也可以调用这个工具来获取。
            该工具使用流程如下：
            1. Agent调用该工具，传入action参数为"search"，查询字符串作为参数
            2. 工具根据查询字符串在当前的工具注册表中搜索相关工具，并返回相关工具的列表供Agent选择。
            3. Agent选择一个工具后，调用该工具，传入action参数为"get"，工具名称作为参数
            4. 工具返回该工具的完整信息（名称、描述、参数等），供Agent调用执行。
            """
        )
        self.get_deferTools_callback = get_deferTools_callback
        self.search_strategy = search_strategy

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        """
        执行工具，根据传入的参数执行相应的操作
        Args:
            parameters (Dict[str, Any]): 工具参数，包含操作类型和查询字符串等信息
            tool_call_id (str): 工具调用ID，用于关联工具调用和工具返回的结果
        """
        action = parameters.get("action")
        query = parameters.get("query", "")
        name = parameters.get("tool_name", "")
        if action == "search":
            tools = self._search_tools(query)
            return Message(role="tool", content=json.dumps(tools, ensure_ascii=False), tool_call_id=tool_call_id)
        elif action == "get":
            tools = self.get_deferTools_callback()
            if name not in tools:   # 这里仅在懒加载工具中搜索，认为普通工具都是已知的，不需要通过这个工具来获取
                return Message(role="tool", content=json.dumps({"error": f"工具 {name} 未找到"}, ensure_ascii=False), tool_call_id=tool_call_id)
            complete_schema = tools[name].to_openai_dict()
            return Message(role="tool", content=json.dumps(complete_schema, ensure_ascii=False), tool_call_id=tool_call_id, metadata={"tool_type": SearchToolsTool, "action": "get"})


    def _search_tools(self, query: str) -> list[Dict[str, str]]:
        defer_tools = self.get_deferTools_callback()
        tool_list = [{"name": tool.name, "description": tool.description} for tool in defer_tools.values()]
        if self.search_strategy == "keyword":
            return self._search_tools_keyword_based(query, tool_list)
        elif self.search_strategy == "subAgent":
            return self._search_tools_subagent_based(query, tool_list)

    def _search_tools_subagent_based(self, query: str, tool_list: list[Dict[str, str]]) -> list[Dict[str, str]]:
        # 基于子Agent的工具搜索：构造一个专门的系统提示，指导子Agent根据工具的名称和描述来判断工具与查询的相关性
        system_prompt = (
            "你是一个工具搜索助手，负责根据用户的查询来判断哪些工具可能相关。"
            "以下是可用工具的列表，每个工具都有一个名称和描述：\n"
        )
        for tool in tool_list:
            system_prompt += f"- {tool['name']}: {tool['description']}\n"
        system_prompt += "请根据用户的查询返回最相关的3个工具。\n"
        system_prompt += "你的输出格式必须为有效的JSON格式，例如：[{\"name\": \"tool1\", \"description\": \"description1\"}, ...]"
        # 创建一个子Agent来处理这个搜索任务
        sub_agent = SearchToolsSubAgent(name="SearchToolsSubAgent", system_prompt=system_prompt)
        response_message = sub_agent.run(query)
        # 解析子Agent的响应，提取出相关工具的名称
        try:
            output_tools = json.loads(response_message.content)

            if isinstance(output_tools, list):
                return output_tools
            else:
                logger.warning("子Agent返回的工具列表格式不正确，预期为列表但得到: %s", output_tools)
                return []
        except Exception as e:
            logger.warning("子Agent返回的工具列表解析失败，确保子Agent的输出是有效的JSON字符串: %s", e)
            return []
            

    def _search_tools_keyword_based(self, query: str, tool_list: list[Dict[str, str]]) -> list[Dict[str, str]]:
        """
        目前实现为简单的TF-IDF向量化检索并仅支持英文！！！
        基于向量化的工具搜索：将查询和工具的名称+描述进行TF-IDF向量化，然后计算查询与每个工具的相似度，返回相似度最高的几个工具
        Args:
            query (str): 用户的查询字符串
            tool_list (list[Dict[str, str]]): 提供的工具列表
        Returns:
            list[Dict[str, str]]: 与查询最相关的工具列表，每个工具包含名称和描述
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            # 准备文档
            documents = [query] + [tool["name"] + ": " + tool["description"] for tool in tool_list]
            # TF-IDF向量化
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)

            # 计算相似度
            query_vector = tfidf_matrix[0:1]
            doc_vector = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vector)
            
            # 获取相似度分数
            similarity_scores = similarities[0]

            # 使用argsort获取排序后的索引
            sorted_indices = np.argsort(similarity_scores)[::-1]  # 降序排列
            
            # 获取前k个最相关的文档
            top_indices = sorted_indices[:3]
            
            # 根据索引检索对应的文档
            retrieved_docs = []
            for idx in top_indices:
                retrieved_docs.append(documents[idx + 1]) 
    
            return retrieved_docs

        except Exception as e:
            logger.exception("向量化检索失败")
            return []

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "action": ToolProperty(
                    type="string",
                    description="要执行的操作（search 或 get）"
                ),
                "query": ToolProperty(
                    type="string",
                    description="搜索查询字符串，仅在action为search时使用"
                ),
                "tool_name": ToolProperty(
                    type="string",
                    description="要获取的工具名称，仅在action为get时使用"
                )
            },
            required=["action"]
        )
    

