"""
TrialAgents Agent单元测试

测试Agent类的基本功能
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import Agent, ClinicianAgent, SupervisorAgent, SimulatorAgent


class TestAgent(unittest.TestCase):
    """测试Agent基类"""
    
    def test_agent_initialization(self):
        """测试Agent初始化"""
        agent = Agent(role="clinician", llm="deepseek-chat")
        self.assertEqual(agent.role, "clinician")
        self.assertEqual(agent.llm, "deepseek-chat")
    
    def test_agent_generation_config(self):
        """测试Agent生成配置"""
        agent = Agent(role="clinician", llm="deepseek-chat")
        config = agent.get_generation_config()
        self.assertIn("temperature", config)
        self.assertIn("max_tokens", config)


class TestSupervisorAgent(unittest.TestCase):
    """测试SupervisorAgent"""
    
    def test_supervisor_initialization(self):
        """测试SupervisorAgent初始化"""
        supervisor = SupervisorAgent(role="supervisor", llm="deepseek-chat")
        self.assertEqual(supervisor.role, "supervisor")
        self.assertEqual(supervisor.llm, "deepseek-chat")
        self.assertEqual(supervisor.current_report, "")
    
    def test_set_basic_info(self):
        """测试设置基本信息"""
        supervisor = SupervisorAgent(llm="deepseek-chat")
        supervisor.set_basic_info("Patient info test")
        self.assertEqual(supervisor.basic_info, "Patient info test")
    
    def test_generate_clinicians_prompt_empty(self):
        """测试生成空clinician prompt"""
        supervisor = SupervisorAgent(llm="deepseek-chat")
        result = supervisor.generate_clinicians_prompt([], is_initial=True)
        self.assertEqual(result, "")


class TestSimulatorAgent(unittest.TestCase):
    """测试SimulatorAgent"""
    
    def test_simulator_initialization(self):
        """测试SimulatorAgent初始化"""
        simulator = SimulatorAgent(
            role="simulator",
            llm="deepseek-chat",
            prediction_horizon=30
        )
        self.assertEqual(simulator.prediction_horizon, 30)
        self.assertEqual(simulator.num_samples, 10)
    
    def test_extract_base_profile(self):
        """测试提取基础档案"""
        simulator = SimulatorAgent(llm="deepseek-chat")
        final_report = {
            "predicted_risk": "high",
            "confidence_score": 0.85,
            "diagnostic_rationale": "test rationale",
        }
        profile = simulator._extract_base_profile(final_report)
        self.assertEqual(profile["predicted_risk"], "high")
        self.assertEqual(profile["confidence_score"], 0.85)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
