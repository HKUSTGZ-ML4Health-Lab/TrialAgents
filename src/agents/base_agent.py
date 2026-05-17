"""
TrialAgents Agent基类

从 m-agents/utils/framework.py 迁移并重构
提供统一的LLM客户端初始化和配置管理
"""

import os
from typing import Dict
from openai import OpenAI
from google import genai

# 从TrialAgents配置导入
from utils.config import LLM_CONFIGS, DEFAULT_LLM


class Agent:
    """
    TrialAgents基础Agent类
    
    所有智能体（ClinicianAgent, SupervisorAgent, SimulatorAgent）的基类。
    提供统一的LLM客户端初始化和API调用接口。
    
    Attributes:
        role: 智能体角色类型（"clinician", "supervisor", "simulator"）
        llm: LLM模型名称
        llm_config: LLM配置字典
        client: OpenAI或Gemini客户端实例
    """
    
    def __init__(
        self,
        role: str = "clinician",
        llm: str = None
    ) -> None:
        """
        初始化Agent基类
        
        Args:
            role: 智能体角色
            llm: LLM模型名称，默认使用配置文件中的DEFAULT_LLM
        """
        self.role = role
        self.llm = llm or DEFAULT_LLM
        
        # 从配置文件获取LLM配置
        if self.llm in LLM_CONFIGS:
            self.llm_config = LLM_CONFIGS[self.llm]
        else:
            # 尝试根据模型名称推断配置
            self.llm_config = self._infer_llm_config(self.llm)
        
        # 初始化LLM客户端
        self.client = self._initialize_client()
    
    def _infer_llm_config(self, llm: str) -> Dict:
        """
        根据模型名称推断配置
        
        如果模型不在预定义配置中，根据名称模式推断配置
        
        Args:
            llm: 模型名称
            
        Returns:
            LLM配置字典
        """
        # 默认配置
        default_config = {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "api_base": "https://api.openai.com/v1",
            "model": llm,
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        
        # 根据模型名称模式推断
        if "deepseek" in llm.lower():
            default_config["api_key"] = os.getenv("DEEPSEEK_API_KEY", "")
            default_config["api_base"] = "https://api.deepseek.com/v1"
        elif "gpt" in llm.lower():
            default_config["api_key"] = os.getenv("OPENAI_API_KEY", "")
            default_config["api_base"] = "https://api.openai.com/v1"
        elif "qwen" in llm.lower() or "doubao" in llm.lower():
            default_config["api_key"] = os.getenv("QWEN_API_KEY", "")
            default_config["api_base"] = "https://dashscope.aliyuncs.com/api/v1"
        elif "gemini" in llm.lower():
            default_config["api_key"] = os.getenv("GEMINI_API_KEY", "")
            default_config["api_base"] = "https://generativelanguage.googleapis.com"
        
        return default_config
    
    def _initialize_client(self):
        """
        初始化LLM客户端
        
        根据模型类型选择OpenAI或Gemini客户端
        
        Returns:
            OpenAI或genai.Client实例
        """
        if 'gemini' in self.llm.lower():
            # Gemini使用专用客户端
            return genai.Client(
                api_key=self.llm_config.get("api_key", ""),
                vertexai=True,  # 可选，优先使用vertexai协议访问，稳定性更高
                http_options={
                    "base_url": self.llm_config.get("api_base", "")
                },
            )
        else:
            # 其他模型使用OpenAI兼容接口
            return OpenAI(
                api_key=self.llm_config.get("api_key", ""),
                base_url=self.llm_config.get("api_base", "")
            )
    
    def get_generation_config(self) -> Dict:
        """
        获取生成配置
        
        子类可以重写此方法以提供特定的生成参数
        
        Returns:
            生成参数字典
        """
        return {
            "temperature": self.llm_config.get("temperature", 0.7),
            "top_p": self.llm_config.get("top_p", 0.9),
            "frequency_penalty": self.llm_config.get("frequency_penalty", 0.0),
            "presence_penalty": self.llm_config.get("presence_penalty", 0.0),
            "max_tokens": self.llm_config.get("max_tokens", 4096),
            "seed": self.llm_config.get("seed", None),
        }
