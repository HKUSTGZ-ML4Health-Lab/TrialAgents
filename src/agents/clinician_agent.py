"""
TrialAgents ClinicianAgent (Clinician Agent)

从 m-agents/utils/framework.py 迁移并重构
负责生成临床综述和参与协作讨论
"""

import os
import json
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_random_exponential, stop_after_attempt

from .base_agent import Agent


class ClinicianAgent(Agent):
    """
    Clinician Agent / Clinician Agent
    
    三名领域专家之一（急诊医学、肿瘤学、综合诊疗），
    负责基于检索到的医学知识生成初步临床综述，
    并在多轮讨论中提供专业意见。
    
    Attributes:
        config: 配置字典
        clinician_index: 医生索引（0, 1, 2对应不同专科）
        retrieval_system: 检索系统实例
        context_builder: 上下文构建器
        latest_analysis: 最新分析结果
    """
    
    def __init__(
        self,
        config: Dict,
        index: int = 0,
        retrieval_system=None,
        role: str = "clinician",
        llm: str = None,
        save_dir: str = "",
        memory_injector=None,  # Memory Bank注入器
    ) -> None:
        """
        初始化ClinicianAgent
        
        Args:
            config: 配置字典，包含dataset_name, task, model_names等
            index: 医生索引（0, 1, 2）
            retrieval_system: 检索系统实例
            role: 角色名称
            llm: LLM模型名称
            save_dir: 保存目录路径
        """
        super().__init__(role, llm)
        
        self.config = config
        self.dataset_name = config.get("dataset_name", "")
        self.task = config.get("task", "")
        self.clinician_index = index
        self.ehr_model_name = config.get("model_names", [""])[index] if index < len(config.get("model_names", [])) else ""
        self.mode = config.get("mode", "mortality_prediction")
        self.retrieval_system = retrieval_system
        self.save_dir = save_dir
        
        # 初始化上下文构建器（从适配器导入）
        # 注意：这里使用适配器从原项目导入
        self.context_builder = self._init_context_builder()
        
        # 加载试验背景信息
        self._load_trial_background()
        
        self.latest_analysis = ""
        self.initial_analysis = ""

        # Memory Bank集成
        self.memory_injector = memory_injector
    
    def _init_context_builder(self):
        """
        初始化上下文构建器
        
        从原项目的适配器导入ContextBuilder
        
        Returns:
            ContextBuilder实例
        """
        try:
            # 尝试从TrialAgents导入（如果已经迁移）
            from utils.adapters import ContextBuilder
        except ImportError:
            # 如果还没有迁移适配器，创建一个简单的占位实现
            ContextBuilder = self._create_placeholder_context_builder()
        
        return ContextBuilder(
            dataset_name=self.dataset_name,
            task=self.task,
            model_name=self.ehr_model_name,
            mode=self.mode
        )
    
    def _create_placeholder_context_builder(self):
        """创建占位上下文构建器"""
        class PlaceholderContextBuilder:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def build(self, patient_id: str) -> Dict:
                return {
                    "main_complaint": "placeholder",
                    "context": "placeholder context"
                }
        
        return PlaceholderContextBuilder
    
    def _load_trial_background(self):
        """
        加载试验背景信息
        
        从corpus/ctgov/{dataset_name}/bg.json加载试验背景
        """
        try:
            # 使用路径工具获取数据目录
            from utils.data_utils.paths import get_raw_data_path
            bg_path = os.path.join(
                get_raw_data_path(),
                'ctgov',
                self.dataset_name,
                'bg.json'
            )
            
            if os.path.exists(bg_path):
                with open(bg_path, 'r') as f:
                    bg_data = json.load(f)
                self.trial_bg = bg_data.get('trial_bg', '')
                self.ctrl_grp = bg_data.get('ctrl_bg', '')
                self.expr_grp = bg_data.get('expr_bg', '')
            else:
                # 如果文件不存在，使用默认值
                self.trial_bg = ""
                self.ctrl_grp = ""
                self.expr_grp = ""
        except Exception as e:
            print(f"Warning: Failed to load trial background: {e}")
            self.trial_bg = ""
            self.ctrl_grp = ""
            self.expr_grp = ""
        
        self.group = self.config.get("group", "experimental")  # control or experimental
    
    def get_role_specific_prompts(self) -> Tuple[str, str, str, str]:
        """
        根据医生索引返回角色特定的prompts
        
        Returns:
            (system_prompt, user_prompt_template, collaboration_system, collaboration_user_template)
        """
        # 尝试从配置加载增强prompt
        try:
            from utils.prompts.template_enhanced_diversity import (
                CLINICIAN_ROLES,
                CLINICIAN_SYSTEM_PROMPTS,
                CLINICIAN_USER_PROMPTS,
                CLINICIAN_COLLABORATION_SYSTEM_PROMPTS,
                CLINICIAN_COLLABORATION_USER_PROMPTS
            )
            
            # role = CLINICIAN_ROLES.get(self.clinician_index, "general_medicine")
            idx = self.clinician_index
            system_prompt = CLINICIAN_SYSTEM_PROMPTS.get(idx, "")
            user_prompt = CLINICIAN_USER_PROMPTS.get(idx, "")
            collab_system = CLINICIAN_COLLABORATION_SYSTEM_PROMPTS.get(idx, "")
            collab_user = CLINICIAN_COLLABORATION_USER_PROMPTS.get(idx, "")
            
        except ImportError:
            # 使用默认prompt
            system_prompt = self._get_default_system_prompt()
            user_prompt = self._get_default_user_prompt()
            collab_system = self._get_default_collaboration_system_prompt()
            collab_user = self._get_default_collaboration_user_prompt()
        
        return system_prompt, user_prompt, collab_system, collab_user
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统prompt"""
        specialties = ["emergency_medicine", "oncology", "general_medicine"]
        specialty = specialties[self.clinician_index] if self.clinician_index < len(specialties) else "general_medicine"
        
        return f"""You are a medical expert specializing in {specialty.replace('_', ' ')}.
Your task is to analyze patient data and provide professional medical assessment.
Provide your analysis in JSON format with the following fields:
- analysis: Your detailed clinical analysis
- logit: Survival probability (0-1)
- conf_logit: Your confidence in this prediction (0-1)"""
    
    def _get_default_user_prompt(self) -> str:
        """获取默认用户prompt"""
        return "Please analyze the following patient data: {context}"
    
    def _get_default_collaboration_system_prompt(self) -> str:
        """获取默认协作系统prompt"""
        return "You are participating in a medical consultation. Review the opinions and provide your assessment."
    
    def _get_default_collaboration_user_prompt(self) -> str:
        """获取默认协作用户prompt"""
        return "Review the following opinions and provide your response: {opinions}"

    def analysis(self, patient_index: int, patient_id: str) -> Tuple[Dict, str, List, int, int]:
        """
        执行临床分析（阶段一）

        基于患者数据和检索到的医学知识生成初步临床综述

        Args:
            patient_index: 患者索引
            patient_id: 患者ID

        Returns:
            (分析结果字典, 医疗上下文, 消息列表, prompt_token数, completion_token数)
        """
        # 构建上下文 - 返回 (basic_context, subcontext, healthcare_context) 三元组
        basic_context, subcontext, healthcare_context = self.context_builder.build(
            patient_index, patient_id
        )

        # 构建 drug group info
        if self.group == 'control':
            drug = self.ctrl_grp
        elif self.group == 'experimental':
            drug = self.expr_grp
        else:
            drug = ""
        group_info = (
            f"\nSince this is a double-blind clinical trial, the patient's group assignment "
            f"(experimental or control) is unknown. Please assume the patient is in the "
            f"{self.group} group receiving the {drug}, and base your subsequent analysis "
            f"on this assumption."
        )

        # 检索相关医学知识（用 rich query）
        doc_context = self.retrieve(self.trial_bg + "\n" + group_info + '\n' + subcontext)

        # 组装最终的上下文文本
        context = self.trial_bg + "\n" + group_info + '\n' + doc_context

        self.patient_info = healthcare_context

        # 使用角色特定的prompts
        system_prompt, user_prompt_template, _, _ = self.get_role_specific_prompts()

        # 兼容 liquid.Template.render() 和 str.format()
        if hasattr(user_prompt_template, 'render'):
            user_prompt = user_prompt_template.render(context=context, hcontext=healthcare_context)
        else:
            user_prompt = user_prompt_template.format(context=context, hcontext=healthcare_context)

        # Memory Bank: 注入相似错误警示
        if self.memory_injector is not None:
            # 构建患者特征文本用于检索
            patient_profile_text = self._build_patient_profile_text(context, patient_id)
            user_prompt = self.memory_injector.inject_for_clinician(
                base_prompt=user_prompt,
                patient_profile_text=patient_profile_text,
                error_types=None,
                k_errors=3,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用LLM
        ans, prompt_token, completion_token = self.invoke(messages=messages)
        
        self.latest_analysis = ans.get("analysis", "")
        self.initial_analysis = ans.get("analysis", "")
        
        return ans, healthcare_context, messages, prompt_token, completion_token
    
    def _retrieve_medical_knowledge(self, context: Dict) -> str:
        """
        检索相关医学知识

        Args:
            context: 患者上下文

        Returns:
            检索到的医学知识文本
        """
        if self.retrieval_system is None:
            return ""

        try:
            query = context.get("main_complaint", "")
            retrieve_texts, _, _ = self.retrieval_system.retrieve(query, k=5)

            contexts = []
            contexts.append("\nThere are some relevant medical documents retrieved from the MSD Manual Professional Edition:")
            for idx, text in enumerate(retrieve_texts):
                contexts.append(f"Document [{idx + 1}] (Title: {text.get('title', 'N/A')}) Content: {text.get('medical_context', '')}")

            return "\n".join(contexts)
        except Exception as e:
            print(f"Warning: Retrieval failed: {e}")
            return ""

    def _build_patient_profile_text(self, context, patient_id: str) -> str:
        """
        构建患者特征文本，用于Memory检索

        Args:
            context: 患者上下文（字符串或字典）
            patient_id: 患者ID

        Returns:
            患者特征描述文本
        """
        group = self.config.get("group", "unknown")
        dataset = self.dataset_name

        # context 可能是字符串或字典
        if isinstance(context, dict):
            features = []
            for key in ["main_complaint", "diagnosis", "treatment", "age", "gender"]:
                if key in context and context[key]:
                    features.append(f"{key}={context[key]}")
            feature_str = ', '.join(features)
        else:
            # context 是字符串，直接截取前200字符作为摘要
            feature_str = str(context)[:200]

        return f"Patient {patient_id} ({dataset}/{group}): {feature_str}"

    def collaboration(self, supervisor_opinion: str, supervisor_report: str, last_review: Dict) -> Tuple[Dict, List, int, int]:
        """
        参与协作讨论（阶段二）
        
        响应Supervisor的意见，提供专业评估
        
        Args:
            supervisor_opinion: Supervisor的意见
            supervisor_report: Supervisor的报告
            last_review: 上一轮的分析结果
            
        Returns:
            (响应字典, 消息列表, prompt_token数, completion_token数)
        """
        context = self.trial_bg
        latest_analysis = f"In the last round, you thought the survival probability of the patient was: {last_review.get('logit', 0)}, with confidence {last_review.get('conf_logit', 0)}.\nYour detailed analysis of the patient is: {self.latest_analysis}"
        
        # 使用角色特定的collaboration prompts
        _, _, collaboration_system, collaboration_user_template = self.get_role_specific_prompts()

        # 兼容 liquid.Template.render() 和 str.format()
        if hasattr(collaboration_user_template, 'render'):
            user_prompt = collaboration_user_template.render(
                context=context,
                analysis=latest_analysis,
                opinion=supervisor_opinion,
                report=supervisor_report
            )
        else:
            user_prompt = collaboration_user_template.format(
                context=context,
                analysis=latest_analysis,
                opinion=supervisor_opinion,
                report=supervisor_report
            )
        
        messages = [
            {"role": "system", "content": collaboration_system},
            {"role": "user", "content": user_prompt}
        ]
        
        ans, prompt_token, completion_token = self.invoke(messages=messages)
        
        # 更新最新分析
        evidences = ans.get("evidences", [])
        evidences_str = "\n".join(evidences) if isinstance(evidences, list) else str(evidences)
        self.latest_analysis = ans.get("reason", "") + evidences_str
        
        return ans, messages, prompt_token, completion_token

    def get_latest_analysis(self, sub_save_dir: str, idx: int) -> Dict:
        """
        获取最新的分析结果
        
        从保存的文件中加载最新的分析结果
        
        Args:
            sub_save_dir: 子保存目录
            idx: 医生索引
            
        Returns:
            最新分析结果字典
        """
        self.latest_reason = None
        self.latest_logit = None
        self.latest_conf = None
        
        # 尝试从多轮讨论文件中加载
        for rnd in range(3, 0, -1):
            file_path = f"{sub_save_dir}/clinician{idx}_round{rnd}_review.json"
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    if "new_logit" in data:
                        self.latest_logit = data["new_logit"]
                        self.latest_conf = data["new_conf_logit"]
                        self.latest_reason = data.get("reason", "")
                        break
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        # 如果没有找到多轮结果，使用初始分析
        if self.latest_logit is None:
            file_path = f"{sub_save_dir}/clinician{idx}_review.json"
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    self.latest_logit = data.get("logit")
                    self.latest_conf = data.get("conf_logit")
                    self.latest_reason = data.get("analysis", "")
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        return {
            "logit": self.latest_logit,
            "conf_logit": self.latest_conf,
            "analysis": self.latest_reason,
        }
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def invoke(self, messages: List[Dict[str, str]]) -> Tuple[Dict, int, int]:
        """
        调用LLM API
        
        Args:
            messages: 消息列表
            
        Returns:
            (响应字典, prompt_token数, completion_token数)
        """
        start = time.time()
        
        try:
            # 获取生成配置
            config = self.get_generation_config()
            
            if 'gemini' not in self.llm.lower():
                # 使用OpenAI兼容接口
                response = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    stream=False,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    frequency_penalty=config["frequency_penalty"],
                    presence_penalty=config["presence_penalty"],
                    max_tokens=config["max_tokens"],
                    seed=config["seed"]
                )
                content = response.choices[0].message.content
                prompt_token = response.usage.prompt_tokens
                completion_token = response.usage.completion_tokens
            else:
                # 使用Gemini接口
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
        print(f"ClinicianAgent-{self.clinician_index} (temp={config['temperature']:.2f}) time cost: {end - start:.2f}s.")
        
        return ans, prompt_token, completion_token

    def retrieve(self, question: str, k: int = 1) -> str:
        """
        检索相关医学文档
        
        Args:
            question: 查询问题
            k: 检索数量
            
        Returns:
            检索结果文本
        """
        if self.retrieval_system is None:
            return ""
        
        try:
            retrieve_texts, _, _ = self.retrieval_system.retrieve(question, k=k)
            contexts = []
            contexts.append("\nThere are some relevant medical documents retrieved from the MSD Manual Professional Edition:")
            for idx, text in enumerate(retrieve_texts):
                contexts.append(f"Document [{idx + 1}] (Title: {text.get('title', 'N/A')}) Content: {text.get('medical_context', '')}")
            return "\n".join(contexts)
        except Exception as e:
            print(f"Warning: Retrieval failed: {e}")
            return ""
