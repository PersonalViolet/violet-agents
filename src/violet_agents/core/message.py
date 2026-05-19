"""
Message类，用于封装messages字段中的每条消息
"""

from typing import Literal, Optional, Any, Dict
from pydantic import BaseModel
from datetime import datetime
MessageRole = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = None

    def __init__(self,
                 content: str,
                 role: MessageRole,
                 name: Optional[str] = None,
                 timestamp: datetime = None,
                 metadata: Optional[Dict[str, Any]] = None):
        if timestamp is None:
            timestamp = datetime.now()
        super().__init__(content=content, 
                         role=role, 
                         timestamp=timestamp, 
                         metadata=metadata,
                         name=name)

    def to_openai_dict(self) -> Dict[str, Any]:
        return self.model_dump(
            exclude={"timestamp", "metadata"},
            exclude_none=True
        )

    def __str__(self) -> str:
        return f"role: {self.role} \n name: {self.name} \n content: {self.content}"