"""配置管理"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    max_history_length: int = 100
    max_sessions: int = 100
    session_default_ttl: Optional[int] = None
    auto_cleanup_sessions: bool = True

    def to_agent_dict(self) -> Dict[str, Any]:
        return self.model_dump(include={"max_history_length"})