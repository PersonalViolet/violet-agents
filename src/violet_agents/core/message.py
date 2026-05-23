"""
Message类，用于封装messages字段中的每条消息
"""

from typing import Literal, Optional, Any, Dict, Union
from pydantic import BaseModel
from datetime import datetime
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
MessageRole = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[Union[list[Dict[str, Any]], list[ChatCompletionMessageFunctionToolCall]]] = None
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None # 适配DeepSeek，记录Agent的思考过程，辅助调试和分析



    def __init__(self,
                 content: str,
                 role: MessageRole,
                 name: Optional[str] = None,
                 timestamp: datetime = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tool_call_id: Optional[str] = None,
                 tool_calls: Optional[Union[list[Dict[str, Any]], list[ChatCompletionMessageFunctionToolCall]]] = None,
                 reasoning_content: Optional[str] = None):
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
                         reasoning_content=reasoning_content)

    def _tool_calls_to_dict(self, tool_calls: Optional[Union[list[Dict[str, Any]], list[ChatCompletionMessageFunctionToolCall]]]) -> Optional[list[Dict[str, Any]]]:
        if tool_calls is None:
            return None
        if isinstance(tool_calls, list) and len(tool_calls) > 0 and isinstance(tool_calls[0], ChatCompletionMessageFunctionToolCall):
            return [tool_call.model_dump() for tool_call in tool_calls]
        return tool_calls

    def to_openai_dict(self) -> Dict[str, Any]:
        return self.model_dump(
            exclude={"timestamp", "metadata"},
            exclude_none=True
        )

    def __str__(self) -> str:
        return f"role: {self.role} \n name: {self.name} \n content: {self.content} \n reasoning_content: {self.reasoning_content}"
    
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
            reasoning_content=chat_message.reasoning_content if hasattr(chat_message, "reasoning_content") else None
        )