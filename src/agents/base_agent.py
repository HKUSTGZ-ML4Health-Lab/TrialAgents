"""
TrialAgents Base Agent

Migrated and refactored from m-agents/utils/framework.py
Provides unified LLM client initialization and configuration management.
"""

import os
from typing import Dict
from openai import OpenAI
from google import genai

from utils.config import LLM_CONFIGS, DEFAULT_LLM


class Agent:
    """
    TrialAgents Base Agent

    Base class for all agents (ClinicianAgent, SupervisorAgent, SimulatorAgent).
    Provides unified LLM client initialization and API call interface.

    Attributes:
        role: Agent role type ("clinician", "supervisor", "simulator")
        llm: LLM model name
        llm_config: LLM configuration dict
        client: OpenAI or Gemini client instance
    """

    def __init__(
        self,
        role: str = "clinician",
        llm: str = None
    ) -> None:
        """
        Initialize the base Agent.

        Args:
            role: Agent role
            llm: LLM model name, defaults to DEFAULT_LLM from config
        """
        self.role = role
        self.llm = llm or DEFAULT_LLM

        if self.llm in LLM_CONFIGS:
            self.llm_config = LLM_CONFIGS[self.llm]
        else:
            self.llm_config = self._infer_llm_config(self.llm)

        self.client = self._initialize_client()

    def _infer_llm_config(self, llm: str) -> Dict:
        """
        Infer LLM configuration from model name.

        If the model is not in predefined configs, infer based on name pattern.

        Args:
            llm: Model name

        Returns:
            LLM configuration dict
        """
        default_config = {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "api_base": "https://api.openai.com/v1",
            "model": llm,
            "max_tokens": 4096,
            "temperature": 0.7,
        }

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
        Initialize the LLM client.

        Selects OpenAI or Gemini client based on model type.

        Returns:
            OpenAI or genai.Client instance
        """
        if 'gemini' in self.llm.lower():
            return genai.Client(
                api_key=self.llm_config.get("api_key", ""),
                vertexai=True,
                http_options={
                    "base_url": self.llm_config.get("api_base", "")
                },
            )
        else:
            return OpenAI(
                api_key=self.llm_config.get("api_key", ""),
                base_url=self.llm_config.get("api_base", "")
            )

    def get_generation_config(self) -> Dict:
        """
        Get the generation configuration.

        Subclasses may override to provide specific generation parameters.

        Returns:
            Generation parameters dict
        """
        return {
            "temperature": self.llm_config.get("temperature", 0.7),
            "top_p": self.llm_config.get("top_p", 0.9),
            "frequency_penalty": self.llm_config.get("frequency_penalty", 0.0),
            "presence_penalty": self.llm_config.get("presence_penalty", 0.0),
            "max_tokens": self.llm_config.get("max_tokens", 4096),
            "seed": self.llm_config.get("seed", None),
        }
