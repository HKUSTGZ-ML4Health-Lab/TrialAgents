"""
TrialAgents SupervisorAgent (Trialist + Supervisor)

从 m-agents/utils/framework.py 迁移并重构
整合Trialist Agent和Supervisor Agent功能：
- Trialist: 整合多医生综述生成动态报告
- Supervisor: 组织协调讨论并生成终版报告
"""

import time
from typing import Dict, List, Tuple
from tenacity import retry, wait_random_exponential, stop_after_attempt

from .base_agent import Agent


class SupervisorAgent(Agent):
    """
    Supervisor Agent / Meta Agent / Supervisor Agent
    
    包含两个核心功能：
    1. Trialist功能: 整合多个ClinicianAgent的综述，生成动态摘要报告
    2. Supervisor功能: 组织协调讨论，整合所有视角，生成终版报告
    
    Attributes:
        current_report: 当前的报告内容
        basic_info: 患者基本信息
    """
    
    def __init__(
        self,
        role: str = "supervisor",
        llm: str = None,
        memory_injector=None,  # Memory Bank注入器
    ) -> None:
        """
        初始化SupervisorAgent

        Args:
            role: 角色名称
            llm: LLM模型名称
            memory_injector: MemoryInjector实例
        """
        super().__init__(role, llm)

        self.current_report = ""
        self.basic_info = None
        self.memory_injector = memory_injector  # Memory Bank集成

        # 初始化prompt（从模板导入）
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化Prompt模板"""
        try:
            # 尝试从模板导入
            from utils.prompts.template import (
                action_system,
                action_user,
                meta_summary_system,
                meta_summary_user,
                collaborative_summary_system,
                collaborative_summary_user,
                revise_logits_system,
                revise_logits_user,
                initial_clinician_summary,
                clinician_low_confidence,
                clinician_moderate_confidence,
                clinician_high_confidence,
                initial_clinician_for_revise,
            )
            
            self.action_system_prompt = action_system
            self.action_user_prompt = action_user
            self.meta_summary_system = meta_summary_system
            self.meta_summary_user = meta_summary_user
            self.collaborative_summary_system = collaborative_summary_system
            self.collaborative_summary_user = collaborative_summary_user
            self.revise_logits_system = revise_logits_system
            self.revise_logits_user = revise_logits_user
            self.initial_clinician_summary = initial_clinician_summary
            self.clinician_low_confidence = clinician_low_confidence
            self.clinician_moderate_confidence = clinician_moderate_confidence
            self.clinician_high_confidence = clinician_high_confidence
            self.initial_clinician_for_revise = initial_clinician_for_revise
            
        except ImportError:
            # 使用默认prompt
            self._init_default_prompts()
    
    def _init_default_prompts(self):
        """初始化默认Prompt"""
        self.action_system_prompt = "You are a medical team coordinator. Analyze clinicians' opinions and decide whether to continue discussion."
        self.action_user_prompt = "Clinicians' opinions:\\n{opinions}\\n\\nDecide: 1 to continue discussion, 0 to finalize report."
        self.meta_summary_system = "You are a medical report writer. Synthesize multiple clinician opinions into a coherent report."
        self.meta_summary_user = "Patient info: {patient_info}\\nClinicians' opinions: {clinician_info}"
        self.collaborative_summary_system = "You are updating a medical report based on new clinician feedback."
        self.collaborative_summary_user = "Current report: {latest_info}\\nNew opinions: {clinician_info}"
        self.revise_logits_system = "You are finalizing medical assessment. Synthesize all clinicians' final opinions."
        self.revise_logits_user = "Clinicians' final opinions: {clinician_info}\\nPrevious report: {latest_info}"
        self.initial_clinician_summary = "Clinician's assessment:\\nDiagnosis: {diagnosis}, Confidence: {conf_logit}\\nAnalysis: {analysis}\\nEvidence: {evidence}"
        self.clinician_low_confidence = "Clinician's response (LOW confidence):\\nReason: {reason}\\nOpinion: {opinion}\\nDocuments: {documents}"
        self.clinician_moderate_confidence = "Clinician's response (MODERATE confidence):\\nReason: {reason}\\nOpinion: {opinion}\\nDocuments: {documents}"
        self.clinician_high_confidence = "Clinician's response (HIGH confidence):\\nReason: {reason}\\nOpinion: {opinion}\\nDocuments: {documents}"
        self.initial_clinician_for_revise = "Clinician {idx}:\\nDiagnosis: {diagnosis}, Confidence: {conf_logit}\\nAnalysis: {analysis}"

    def set_basic_info(self, patient_info: str) -> None:
        """
        设置患者基本信息

        Args:
            patient_info: 患者信息文本
        """
        self.basic_info = patient_info

    def _render_template(self, template, **kwargs) -> str:
        """
        渲染模板，兼容 liquid.Template.render() 和 str.format()
        """
        if hasattr(template, 'render'):
            return template.render(**kwargs)
        else:
            return template.format(**kwargs)

    def generate_clinicians_prompt(self, clinician_responses: List[Dict], is_initial: bool = False) -> str:
        """
        生成医生意见的prompt文本
        
        Args:
            clinician_responses: clinician响应列表
            is_initial: 是否为初始综述
            
        Returns:
            格式化的医生意见文本
        """
        responses = ""
        for idx, res in enumerate(clinician_responses):
            if is_initial:
                # 初始综述格式
                evidence = res.get('evidences', [])
                if evidence and not isinstance(evidence[0], str):
                    evidence = list(map(str, evidence))
                evidence_str = '\\n'.join(evidence) if isinstance(evidence, list) else str(evidence)
                
                response = self._render_template(self.initial_clinician_summary,
                    diagnosis=res.get('logit', 0),
                    conf_logit=res.get('conf_logit', 0),
                    analysis=res.get('analysis', ''),
                    evidence=evidence_str
                )
            else:
                # 协作讨论格式
                documents = '\\n'.join(res.get('evidences', [])) if isinstance(res.get('evidences'), list) else str(res.get('evidences', ''))
                
                confidence = res.get('confidence', 1)
                if confidence == 1 or 'confidence' not in res:
                    response = self._render_template(self.clinician_low_confidence,
                        reason=res.get('reason', ''),
                        opinion=res.get('answer', ''),
                        documents=documents
                    )
                elif confidence == 2:
                    response = self._render_template(self.clinician_moderate_confidence,
                        reason=res.get('reason', ''),
                        opinion=res.get('answer', ''),
                        documents=documents
                    )
                elif confidence == 3:
                    response = self._render_template(self.clinician_high_confidence,
                        reason=res.get('reason', ''),
                        opinion=res.get('answer', ''),
                        documents=documents
                    )
                else:
                    print(f"Clinician Confidence Error: {confidence}!")
                    response = f"Clinician's response: {res}"
            
            response = f"Clinician {idx}'s statement is as follows:\\n{response}\n"
            responses = responses + response

        return responses

    def extract_action_response(self, response: Dict) -> int:
        """
        提取动作响应
        
        Args:
            response: LLM响应字典
            
        Returns:
            1 继续讨论, 0 结束讨论
        """
        action = response.get('action', '').lower()
        if "yes" in action or "continue" in action or "1" in action:
            return 1
        elif "no" in action or "finalize" in action or "0" in action:
            return 0
        else:
            # 默认继续
            print(f"Warning: Invalid action response: {action}. Defaulting to continue.")
            return 1

    def next_action(self, clinician_responses: List[Dict]) -> Tuple[int, Dict, List, int, int]:
        """
        决定下一步动作
        
        分析医生们的响应，决定是否继续讨论
        
        Args:
            clinician_responses: clinician响应列表，每个包含：
                - answer: agree/disagree
                - confidence: 0-1
                - reason: 理由
                - evidences: 证据
                
        Returns:
            (action, response_dict, messages, prompt_tokens, completion_tokens)
            action: 1 继续循环, 0 结束循环
        """
        action_user_prompt = self.generate_clinicians_prompt(clinician_responses)
        messages = [
            {"role": "system", "content": self.action_system_prompt},
            {"role": "user", "content": self._render_template(self.action_user_prompt, opinions=action_user_prompt)}
        ]
        ans, prompt_token, completion_token = self.invoke(messages)
        action = self.extract_action_response(ans)
        return action, ans, messages, prompt_token, completion_token

    def summary(self, clinician_responses: List[Dict], is_initial: bool = False) -> Tuple[Dict, List, int, int]:
        """
        生成/更新报告（Trialist功能）
        
        整合医生们的意见，生成动态摘要报告
        
        Args:
            clinician_responses: clinician响应列表
            is_initial: 是否为初始综述
            
        Returns:
            (response_dict, messages, prompt_tokens, completion_tokens)
        """
        clinicians_answer_prompt = self.generate_clinicians_prompt(clinician_responses, is_initial)

        if is_initial:
            # 初始综述
            messages = [
                {"role": "system", "content": self.meta_summary_system},
                {"role": "user", "content": self._render_template(self.meta_summary_user,
                    patient_info=self.basic_info,
                    clinician_info=clinicians_answer_prompt
                )}
            ]
            ans, prompt_token, completion_token = self.invoke(messages)
            self.current_report = ans.get('answer', '') + "\\n" + ans.get('report', '')
            return ans, messages, prompt_token, completion_token
        else:
            # 更新报告（协作阶段）
            messages = [
                {"role": "system", "content": self.collaborative_summary_system},
                {"role": "user", "content": self._render_template(self.collaborative_summary_user,
                    clinician_info=clinicians_answer_prompt,
                    latest_info=self.current_report
                )}
            ]
            ans, prompt_token, completion_token = self.invoke(messages)
            action = self.extract_action_response(ans)
            self.current_report = ans.get('answer', '') + "\\n" + ans.get('report', '')
            return action, ans, messages, prompt_token, completion_token

    def generate_clinicians_initial_prompt(self, clinician_responses: List[Dict]) -> str:
        """
        生成初始clinician意见的prompt（用于revise_logits）

        Args:
            clinician_responses: clinician响应列表

        Returns:
            格式化的prompt文本
        """
        responses = ""
        for idx, clinician in enumerate(clinician_responses):
            prompt = self._render_template(self.initial_clinician_for_revise,
                idx=idx,
                diagnosis=clinician.get('logit', 0),
                conf_logit=clinician.get('conf_logit', 0),
                analysis=clinician.get('analysis', '')
            )
            if idx < len(clinician_responses) - 1:
                response = f"Clinician {idx}'s statement is as follows:\\n{prompt}\n"
            else:
                response = f"Clinician {idx}'s statement is as follows:\\n{prompt}"
            responses = responses + response
        return responses

    def revise_logits(self, clinician_initial_responses: List[Dict], patient_profile_text: str = None) -> Tuple[Dict, List, int, int]:
        """
        修正并整合最终预测（Supervisor功能）

        基于所有医生的最终意见，生成终版报告

        Args:
            clinician_initial_responses: 医生的最终分析结果
            patient_profile_text: 患者特征文本（用于Memory检索）

        Returns:
            (final_prediction_dict, messages, prompt_tokens, completion_tokens)
        """
        clinicians_answer_prompt = self.generate_clinicians_initial_prompt(clinician_initial_responses)
        user_prompt = self._render_template(self.revise_logits_user,
            clinician_info=clinicians_answer_prompt,
            latest_info=self.current_report
        )

        # Memory Bank: 注入相似错误和成功案例
        if self.memory_injector is not None and patient_profile_text:
            user_prompt = self.memory_injector.inject_for_supervisor(
                base_prompt=user_prompt,
                patient_profile_text=patient_profile_text,
                k_errors=2,
                k_successes=2,
            )

        messages = [
            {"role": "system", "content": self.revise_logits_system},
            {"role": "user", "content": user_prompt}
        ]
        ans, prompt_token, completion_token = self.invoke(messages)
        return ans, messages, prompt_token, completion_token

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def invoke(self, messages: List[Dict[str, str]]) -> Tuple[Dict, int, int]:
        """
        调用LLM API
        
        SupervisorAgent使用更保守的参数设置，确保一致性和准确性
        
        Args:
            messages: 消息列表
            
        Returns:
            (response_dict, prompt_tokens, completion_tokens)
        """
        start = time.time()
        
        try:
            # SupervisorAgent使用更保守的参数
            if 'gemini' not in self.llm.lower():
                response = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    stream=False,
                    temperature=0.3,        # 较低温度，确保一致性
                    top_p=0.9,           # 适中的top_p
                    frequency_penalty=0.0,  # 不惩罚重复，保持专业性
                    presence_penalty=0.0,   # 不强制引入新话题
                    max_tokens=2048,
                    seed=None
                )
                content = response.choices[0].message.content
                prompt_token = response.usage.prompt_tokens
                completion_token = response.usage.completion_tokens
            else:
                # Gemini接口
                response = self.client.models.generate_content(
                    model=self.llm,
                    contents=[messages[0]['content'], messages[1]['content']],
                )
                content = response.text
                prompt_token = response.usage_metadata.prompt_token_count
                completion_token = response.usage_metadata.candidates_token_count
            
            # 解析JSON响应
            from .json_parser import extract_and_parse_json
            ans = extract_and_parse_json(content)
            
        except Exception as e:
            raise e
        
        end = time.time()
        print(f"SupervisorAgent (temp=0.3) time cost: {end - start:.2f}s.")
        
        return ans, prompt_token, completion_token

    def get_generation_config(self) -> Dict:
        """
        获取生成配置（SupervisorAgent使用保守参数）
        
        Returns:
            生成参数字典
        """
        return {
            "temperature": 0.3,  # 较低温度，确保一致性
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 2048,
            "seed": None,
        }
