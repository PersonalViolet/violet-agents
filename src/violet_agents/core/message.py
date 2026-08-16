"""
Message类，用于封装messages字段中的每条消息
"""

from typing import Literal, Optional, Any, Dict, Union
from pydantic import BaseModel
from datetime import datetime
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
MessageRole = Literal["system", "user", "assistant", "tool"]

class Usage(BaseModel):
    """LLM 返回的 token 用量信息。

    兼容 OpenAI 标准字段，同时适配 DeepSeek 特有的 prompt 缓存命中字段：
    - completion_tokens: 本次生成的 token 数
    - prompt_tokens: 本次请求的 prompt token 数
    - total_tokens: 总 token 数（prompt + completion）
    - prompt_cache_hit_tokens: prompt 缓存命中的 token 数（DeepSeek 特有）
    - prompt_cache_miss_tokens: prompt 缓存未命中的 token 数（DeepSeek 特有）
    """
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_cache_hit_tokens: Optional[int] = None
    prompt_cache_miss_tokens: Optional[int] = None


class Message(BaseModel):
    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[Union[list[Dict[str, Any]], list[ChatCompletionMessageFunctionToolCall]]] = None
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None # 适配DeepSeek，记录Agent的思考过程，辅助调试和分析
    usage: Optional[Usage] = None


    def __init__(self,
                 content: str,
                 role: MessageRole,
                 name: Optional[str] = None,
                 timestamp: datetime = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tool_call_id: Optional[str] = None,
                 tool_calls: Optional[Union[list[Dict[str, Any]], list[ChatCompletionMessageFunctionToolCall]]] = None,
                 reasoning_content: Optional[str] = None,
                 usage: Optional[Usage] = None):
        if timestamp is None:
            timestamp = datetime.now()
        tool_calls = self._tool_calls_to_dict(tool_calls)
        super().__init__(content=content,
                         role=role,
                         timestamp=timestamp,
                         metadata=metadata,
                         name=name,
                         tool_call_id=tool_call_id,
                         tool_calls=tool_calls,
                         reasoning_content=reasoning_content,
                         usage=usage)

    def _tool_calls_to_dict(self, tool_calls: Optional[Union[list[Dict[str, Any]], list[ChatCompletionMessageFunctionToolCall]]]) -> Optional[list[Dict[str, Any]]]:
        if tool_calls is None:
            return None
        if isinstance(tool_calls, list) and len(tool_calls) > 0 and isinstance(tool_calls[0], ChatCompletionMessageFunctionToolCall):
            return [tool_call.model_dump() for tool_call in tool_calls]
        return tool_calls

    def to_openai_dict(self) -> Dict[str, Any]:
        return self.model_dump(
            exclude={"timestamp", "metadata", "usage"},
            exclude_none=True
        )

    def __str__(self) -> str:
        return f"role: {self.role} \n name: {self.name} \n content: {self.content} \n reasoning_content: {self.reasoning_content} \n usage: {self.usage}"

    @staticmethod
    def from_chat_completion_message(chat_message: ChatCompletionMessage) -> 'Message':
        """
        将OpenAI API返回的ChatCompletionMessage对象转换为Message对象
        Args:
            chat_message (ChatCompletionMessage): OpenAI API返回的消息对象，包含role、content、tool_calls等信息
        Returns:
            Message: 转换后的Message对象，包含role、content、tool_calls等信息
        """
        return Message(
            content=chat_message.content,
            role=chat_message.role,
            name=chat_message.name if hasattr(chat_message, "name") else None,
            tool_call_id=chat_message.tool_calls[0].id if hasattr(chat_message, "tool_calls") and chat_message.tool_calls else None,
            tool_calls=[tool_call.model_dump() for tool_call in chat_message.tool_calls] if hasattr(chat_message, "tool_calls") and chat_message.tool_calls else None,
            reasoning_content=chat_message.reasoning_content if hasattr(chat_message, "reasoning_content") else None,
        )

    @staticmethod
    def from_chat_completion(chat_completion: ChatCompletion) -> 'Message':
        """
        将OpenAI API返回的完整ChatCompletion对象转换为Message对象，并附带token用量信息

        Args:
            chat_completion (ChatCompletion): OpenAI API返回的完整响应，包含choices与usage

        Returns:
            Message: 转换后的Message对象，其usage字段携带token用量信息
        """
        message = Message.from_chat_completion_message(chat_completion.choices[0].message)
        message.usage = Message._extract_usage(chat_completion)
        return message

    @staticmethod
    def _extract_usage(chat_completion: ChatCompletion) -> Optional[Usage]:
        """
        从ChatCompletion响应中提取token用量信息

        标准字段（completion_tokens/prompt_tokens/total_tokens）直接从usage中读取；
        DeepSeek特有的prompt_cache_hit_tokens与prompt_cache_miss_tokens存储在
        usage.model_extra中（openai SDK对其无法识别的字段会保留在extra里）。

        Args:
            chat_completion (ChatCompletion): OpenAI API返回的完整响应

        Returns:
            Optional[Usage]: 提取到的用量信息，若无usage则为None
        """
        usage = chat_completion.usage
        if usage is None:
            return None
        return Usage(
            completion_tokens=usage.completion_tokens,
            prompt_tokens=usage.prompt_tokens,
            total_tokens=usage.total_tokens,
            prompt_cache_hit_tokens=usage.model_extra.get("prompt_cache_hit_tokens") if usage.model_extra else None,
            prompt_cache_miss_tokens=usage.model_extra.get("prompt_cache_miss_tokens") if usage.model_extra else None,
        )