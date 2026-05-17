"""
TrialAgents智能体模块

包含四个核心智能体：
- ClinicianAgent: 三个领域专家（急诊医学、肿瘤学、综合诊疗）
- SupervisorAgent: 包含Trialist和Supervisor功能
- SimulatorAgent: 数字孪生生成

迁移说明：
本模块代码从 m-agents/utils/framework.py 迁移至此
"""

from .base_agent import Agent
from .clinician_agent import ClinicianAgent
from .supervisor_agent import SupervisorAgent
from .simulator_agent import SimulatorAgent
from .json_parser import extract_and_parse_json

__all__ = [
    'Agent',
    'ClinicianAgent',
    'SupervisorAgent',
    'SimulatorAgent',
    'extract_and_parse_json',
]
