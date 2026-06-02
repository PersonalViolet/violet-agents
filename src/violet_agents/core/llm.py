"""
统一LLM接口 - 支持OpenAI-API

环境变量名解释：
- LLM_API_KEY: LLM服务密钥，优先级低于各自服务的专用环境变量
- LLM_BASE_URL: LLM服务地址，优先级低于各自服务的专用环境变量
- LLM_MODEL_NAME: 模型名称，优先级低于各自服务的专用环境变量

- Deepseek专用环境变量：
    - DEEPSEEK_API_KEY  
    - DEEPSEEK_BASE_URL     
    - DEEPSEEK_MODEL_NAME

- Modelscope专用环境变量：
    - MODELSCOPE_API_KEY
    - MODELSCOPE_BASE_URL
    - MODELSCOPE_MODEL_NAME

在provider有效情况下，若参数、环境变量读取不到model名称，则使用模型列表：
- deepseek: deepseek-v4-flash
- modelscope: qwen3.5-flash
"""
import os
from typing import Literal, Optional, Iterator, Dict, Any, List, Union
from collections import deque
from openai import OpenAI
from .exceptions import VioletAgentException
from openai.types.chat import ChatCompletion
from .message import Message


SUPPORTED_PROVIDERS = Literal[
    "deepseek", "modelscope", "auto"
]
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "modelscope": "qwen3.5-flash",
}

TOOL_CHOICE = Literal[
    'auto', 'none', 'required'
]

class VioletAgentsLLM:

    def __init__(self,
                 model: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 provider: Optional[SUPPORTED_PROVIDERS] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 timeout: Optional[int] = None,
                 **kwargs):

        """
        初始化客户端，支持从环境变量中读取配置，优先级为：参数 > 环境变量 > 默认值
        支持通过检测provider判断加载api_key, base_url对应的环境变量名，model为空时为model赋默认值。若不提供provider则通过检测base_url与api_key的值来确认provider。
        
        Args:
            model: 模型名称，若未提供则从环境变量读取，环境变量不存在时根据provider自动设置
            base_url: LLM服务地址，若未提供则从环境变量读取
            api_key: LLM服务密钥，若未提供则从环境变量读取
            provider: LLM服务提供商，若未提供则通过环境变量名LLM_API_KEY和LLM_BASE_URL自动检测，若未提供则使用通用配置
            temperature: 模型温度，默认为0.7
            max_tokens: 最大token数，默认为None
            timeout: 请求超时时间，默认为None

        example:
        ```python
        llmA = VioletAgentsLLM(provider="deepseek")
        #使用该构造函数时，你需要确保环境变量LLM_BASE_URL存在，若你使用的服务商不支持本项目的自动检测功能，确保环境变量LLM_API_KEY和LLM_MODEL_NAME存在；若你使用的服务商在本项目中支持自动检测功能，确保环境变量${PROVIDER}_API_KEY或LLM_API_KEY存在
        llmB = VioletAgentsLLM() 
        ``` 
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider or self._auto_detect_provider(base_url)
        self.api_key, self.base_url, self.model = self._revolve_credentials(base_url, api_key, model)
        self.timeout = timeout
        self.kwargs = kwargs
        if not self.api_key or not self.base_url:
            raise VioletAgentException("LLM服务配置错误，请检查环境变量配置")
        if not self.model:
            self.model = self._get_default_model()
            if not self.model:
                raise VioletAgentException("未找到模型名称，请检查环境变量配置或手动指定模型名称")
        
        self.client = self._create_client()

    def _create_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _auto_detect_provider(self, base_url: Optional[str] ) -> SUPPORTED_PROVIDERS:
        """
        自动检测LLM服务提供商

        检查逻辑：
        1. 根据base_url格式判断
        2. 默认返回通用配置
        """
        # 1. 根据base_url格式判断
        actual_base_url = base_url or os.getenv('LLM_BASE_URL')
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if 'api.deepseek.com' in base_url_lower:
                return 'deepseek'
            elif 'dashscope.aliyuncs.com' in base_url_lower:
                return 'modelscope'
        # 2. 默认返回通用配置
        return "auto"
    
    def _revolve_credentials(self, base_url: Optional[str], api_key: Optional[str], model: str) -> tuple[str, str, str]:
        """
        根据provider和base_url，api_key进行参数解析
        """
        revolve_api_key: str
        revolve_base_url: str
        revolve_model: str
        if self.provider == 'deepseek':
            revolve_api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY')
            revolve_base_url = base_url or os.getenv('DEEPSEEK_BASE_URL') or os.getenv('LLM_BASE_URL')
            revolve_model = model or os.getenv('DEEPSEEK_MODEL_NAME') or os.getenv('LLM_MODEL_NAME')
        elif self.provider == 'modelscope':
            revolve_api_key = api_key or os.getenv('MODELSCOPE_API_KEY') or os.getenv('LLM_API_KEY')
            revolve_base_url = base_url or os.getenv('MODELSCOPE_BASE_URL') or os.getenv('LLM_BASE_URL')
            revolve_model = model or os.getenv('MODELSCOPE_MODEL_NAME') or os.getenv('LLM_MODEL_NAME')
        else:
            revolve_api_key = api_key or os.getenv('LLM_API_KEY')
            revolve_base_url = base_url or os.getenv('LLM_BASE_URL')
            revolve_model = model or os.getenv('LLM_MODEL_NAME')
        return revolve_api_key, revolve_base_url, revolve_model

    def _get_default_model(self) -> Optional[str]:
        """
        根据provider返回默认模型名称

        该方法会查询DEFAULT_MODELS字典，根据当前实例的provider值返回相应的默认模型名称。
        如果provider没有在DEFAULT_MODELS中定义，则返回None。
        """
        return DEFAULT_MODELS.get(self.provider)
        
            
    def chat(self, 
             messages: Union[list[dict[str, Any]], deque[Message]], 
             tools: Optional[List[Dict[str, Any]]] = None, 
             tool_choice: Optional[TOOL_CHOICE] = 'none',
             **kwargs) -> ChatCompletion:
        """
        调用LLM服务进行对话，返回完整响应

        Args:
            messages: 类型支持list[dict[str, Any]]或deque[Message]，对话消息列表，每个消息包含role和content字段
            tools: 工具列表，每个工具包含name和description字段
            tool_choice: 工具选择策略

        Returns:
            LLM服务返回的响应
        """

        if isinstance(messages, deque):
            messages = list(messages)
        if messages and isinstance(messages[0], Message):
            messages = [message.to_openai_dict() for message in messages]

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            **kwargs
        )
        
        
    
