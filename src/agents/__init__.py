"""
TrialAgents Agent Module

Core agents:
- ClinicianAgent: Domain experts (emergency medicine, oncology, general practice)
- SupervisorAgent: Trialist and Supervisor functionality
- SimulatorAgent: Digital twin generation

Migrated from m-agents/utils/framework.py
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
