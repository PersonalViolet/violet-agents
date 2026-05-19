"""
异常体系
"""
class VioletAgentException(Exception):
    """VioletAgent的基础异常类"""
    pass

class LLMException(VioletAgentException):
    """LLM相关的异常"""
    pass

class AgentException(VioletAgentException):
    """Agent相关的异常"""
    pass

class ConfigException(VioletAgentException):
    """Config相关的异常"""
    pass

class ToolException(VioletAgentException):
    """Tool相关的异常"""
    pass
