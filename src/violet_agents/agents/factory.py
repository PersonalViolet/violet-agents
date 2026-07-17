"""agent 工厂函数

用于创建不同类型的 agent 实例。
"""

from typing import Optional, Literal, TYPE_CHECKING
from ..core import VioletAgentsLLM, Config
from ..core import Agent
if TYPE_CHECKING:
    from ..core import ToolRegistry

AgentMode  = Literal["react", "simple"]

def create_agent(
        agent_type: AgentMode,
        llm: VioletAgentsLLM,
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[Config] = None,
        system_prompt: Optional[str] = None
) -> Agent:
    """创建 agent 实例

    Args:
        agent_type (AgentMode): agent 类型，支持 "react" 和 "simple"
        llm (VioletAgentsLLM): LLM 实例
        tool_registry (Optional[ToolRegistry], optional): 工具注册表实例. Defaults to None.
        config (Optional[Config], optional): 配置实例. Defaults to None.
        system_prompt (Optional[str], optional): 系统提示语. Defaults to None.

    Returns:
        Agent: 创建的 agent 实例
    """
    agent_type = agent_type.lower()
    if agent_type == "react":
        from .react_agent import ReactAgent
        return ReactAgent(llm=llm, tool_registry=tool_registry, config=config, system_prompt=system_prompt)
    elif agent_type == "simple":
        from .simple_agent import SimpleAgent
        return SimpleAgent(llm=llm, tool_registry=tool_registry, config=config, system_prompt=system_prompt)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    

