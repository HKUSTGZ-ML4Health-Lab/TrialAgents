"""
TrialAgents 协作引擎

从 m-agents/utils/framework.py 迁移并重构
负责管理多智能体之间的多轮协作讨论流程
"""

import json
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..agents.supervisor_agent import SupervisorAgent  # pyright: ignore[reportMissingImports]
from ..agents.clinician_agent import ClinicianAgent  # pyright: ignore[reportMissingImports]


class CollaborationEngine:
    """
    协作引擎 / Collaboration Engine
    
    管理多轮协作讨论流程，协调SupervisorAgent和多个ClinicianAgent之间的交互。
    
    工作流程：
    1. 初始化：接收初始Supervisor意见和报告
    2. 多轮讨论：每轮收集Clinician反馈，Supervisor整合
    3. 终止判断：根据Supervisor的action决定是否继续或结束
    4. 输出：最终的报告和token统计
    
    Attributes:
        supervisor_agent: SupervisorAgent实例
        clinician_agents: ClinicianAgent列表
        supervisor_opinion: 当前Supervisor意见
        supervisor_report: 当前Supervisor报告
        clinician_responses: clinician响应列表
        clinician_num: clinician数量
        max_round: 最大讨论轮数
        now_round: 当前轮数
        save_dir: 保存目录
    """
    
    def __init__(
        self,
        supervisor_agent: SupervisorAgent,
        clinician_agents: List[ClinicianAgent],
        supervisor_opinion: str,
        supervisor_report: str,
        save_dir: str,
        clinician_num: int = 3,
        max_round: int = 3
    ):
        """
        初始化协作引擎
        
        Args:
            supervisor_agent: SupervisorAgent实例
            clinician_agents: ClinicianAgent列表
            supervisor_opinion: 初始Supervisor意见
            supervisor_report: 初始Supervisor报告
            save_dir: 保存目录路径
            clinician_num: clinician数量
            max_round: 最大讨论轮数
        """
        self.supervisor_agent = supervisor_agent
        self.clinician_agents = clinician_agents
        self.supervisor_opinion = supervisor_opinion
        self.supervisor_report = supervisor_report
        self.save_dir = save_dir
        self.clinician_num = clinician_num
        self.max_round = max_round
        
        self.clinician_responses = []
        self.now_round = 0
    
    def collaborate(self) -> Tuple[str, int, int, int]:
        """
        执行多轮协作讨论
        
        核心方法，执行阶段二的协作流程：
        1. 每轮并行收集所有Clinician的反馈
        2. Supervisor整合反馈并生成新报告
        3. Supervisor决定是继续还是结束
        4. 达到max_round或Supervisor决定结束则退出
        
        Returns:
            (final_report, final_round, prompt_tokens, completion_tokens)
            final_report: 最终报告文本
            final_round: 实际进行的轮数
            prompt_tokens: 总prompt token数
            completion_tokens: 总completion token数
        """
        # 确保初始状态正确
        assert len(self.clinician_responses) == 0 and self.now_round == 0, \
            "Collaboration already started or invalid state"
        
        prompt_tokens = 0
        completion_tokens = 0
        
        # 多轮讨论循环
        while self.now_round < self.max_round:
            self.now_round += 1
            print(f"\\n{'='*50}")
            print(f"Starting Round {self.now_round}/{self.max_round}")
            print(f"{'='*50}")
            
            # 清空上一轮响应
            self.clinician_responses = []
            
            # 并行处理所有ClinicianAgent
            def process_clinician(clinician_agent: ClinicianAgent, idx: int) -> Tuple[int, int]:
                """
                处理单个ClinicianAgent的协作
                
                Args:
                    clinician_agent: ClinicianAgent实例
                    idx: clinician索引
                    
                Returns:
                    (prompt_token, completion_token)
                """
                # 读取上一轮分析结果
                review_file = f"{self.save_dir}/clinician{idx + 1}_review.json"
                try:
                    with open(review_file, 'r') as f:
                        last_review = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {review_file}: {e}")
                    last_review = {'logit': 0.5, 'conf_logit': 0.5}
                
                # 执行协作讨论
                response, messages, prompt_token, completion_token = clinician_agent.collaboration(
                    supervisor_opinion=self.supervisor_opinion,
                    supervisor_report=self.supervisor_report,
                    last_review=last_review
                )

                # 将响应添加到共享列表，供Supervisor整合
                self.clinician_responses.append(response)

                # 保存响应
                round_file = f"{self.save_dir}/clinician{idx + 1}_round{self.now_round}_review.json"
                with open(round_file, "w") as f:
                    json.dump(response, f, indent=2)
                
                # messages_file = f"{self.save_dir}/clinician{idx + 1}_round{self.now_round}_review_messages.json"
                # with open(messages_file, "w") as f:
                #     json.dump(messages, f, indent=2)
                
                # 保存user prompt
                if len(messages) > 1:
                    prompt_file = f"{self.save_dir}/clinician{idx + 1}_round{self.now_round}_review_userprompt.txt"
                    with open(prompt_file, "w") as f:
                        f.write(messages[1].get("content", ""))
                
                return prompt_token, completion_token
            
            # 并行执行所有ClinicianAgent
            with ThreadPoolExecutor(max_workers=self.clinician_num) as executor:
                futures = [
                    executor.submit(process_clinician, clinician_agent, i)
                    for i, clinician_agent in enumerate(self.clinician_agents)
                ]
                
                # 收集结果
                for future in futures:
                    try:
                        prompt_token, completion_token = future.result()
                        prompt_tokens += prompt_token
                        completion_tokens += completion_token
                    except Exception as e:
                        print(f"Error in doctor process: {e}")
            
            # Supervisor整合所有Clinician的反馈
            print(f"\\nSupervisorAgent is summarizing round {self.now_round}...")
            action, summary_content, messages, prompt_token, completion_token = \
                self.supervisor_agent.summary(self.clinician_responses, is_initial=False)
            
            # 保存Supervisor的总结
            summary_file = f"{self.save_dir}/supervisor_round{self.now_round}_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary_content, f, indent=2)
            
            # 保存messages
            messages_file = f"{self.save_dir}/supervisor_round{self.now_round}_summary_messages.json"
            with open(messages_file, "w") as f:
                json.dump(messages, f, indent=2)
            
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            
            # 更新Supervisor意见和报告
            self.supervisor_opinion = summary_content.get("answer", self.supervisor_opinion)
            self.supervisor_report = summary_content.get("report", self.supervisor_report)
            
            print(f"\\nRound {self.now_round} completed.")
            print(f"Action: {'Continue' if action == 1 else 'Finalize'}")
            
            # 检查是否结束讨论
            if action == 0:
                print(f"\\n{'='*50}")
                print("Consensus reached! Finalizing report...")
                print(f"{'='*50}")
                break
            else:
                print("Continuing to next round...")
        
        # 检查是否达到最大轮数仍未达成共识
        if self.now_round == self.max_round:
            print(f"\\n{'='*50}")
            print(f"Warning: No agreement reached after {self.max_round} rounds.")
            print("Using last round's report as final.")
            print(f"{'='*50}")
        
        # 重置轮数计数器
        final_round = self.now_round
        self.now_round = 0
        
        return self.supervisor_report, final_round, prompt_tokens, completion_tokens
    
    def get_discussion_summary(self) -> Dict:
        """
        获取讨论摘要
        
        返回多轮讨论的统计信息和关键结果
        
        Returns:
            讨论摘要字典
        """
        return {
            "total_rounds": self.now_round,
            "max_rounds": self.max_round,
            "clinician_num": self.clinician_num,
            "final_opinion": self.supervisor_opinion,
            "discussion_complete": self.now_round < self.max_round or self.now_round == 0,
        }
    
    def save_intermediate_results(self) -> None:
        """
        保存中间结果
        
        将所有中间结果保存到指定目录
        """
        # 保存当前状态
        state_file = f"{self.save_dir}/collaboration_state.json"
        state = {
            "now_round": self.now_round,
            "max_round": self.max_round,
            "supervisor_opinion": self.supervisor_opinion,
            "supervisor_report": self.supervisor_report,
            "clinician_responses_count": len(self.clinician_responses),
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)


# 为了向后兼容，保留Collaboration别名
Collaboration = CollaborationEngine
