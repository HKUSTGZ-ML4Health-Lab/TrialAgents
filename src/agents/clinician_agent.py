"""
TrialAgents ClinicianAgent

Migrated and refactored from m-agents/utils/framework.py
Responsible for generating clinical reviews and participating in collaborative discussion.
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
    Clinician Agent

    One of three domain experts (emergency medicine, oncology, general practice),
    responsible for generating initial clinical reviews based on retrieved medical
    knowledge and providing expert opinions during multi-round discussion.

    Attributes:
        config: Configuration dict
        clinician_index: Clinician index (0, 1, 2 for different specialties)
        retrieval_system: Retrieval system instance
        context_builder: Context builder
        latest_analysis: Latest analysis result
    """

    def __init__(
        self,
        config: Dict,
        index: int = 0,
        retrieval_system=None,
        role: str = "clinician",
        llm: str = None,
        save_dir: str = "",
        memory_injector=None,
    ) -> None:
        """
        Initialize ClinicianAgent.

        Args:
            config: Config dict with dataset_name, task, model_names, etc.
            index: Clinician index (0, 1, 2)
            retrieval_system: Retrieval system instance
            role: Role name
            llm: LLM model name
            save_dir: Save directory path
            memory_injector: Memory Bank injector
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

        self.context_builder = self._init_context_builder()

        self._load_trial_background()

        self.latest_analysis = ""
        self.initial_analysis = ""

        self.memory_injector = memory_injector

    def _init_context_builder(self):
        """
        Initialize the context builder.

        Uses ContextBuilder from adapters.

        Returns:
            ContextBuilder instance
        """
        try:
            from utils.adapters import ContextBuilder
        except ImportError:
            ContextBuilder = self._create_placeholder_context_builder()

        return ContextBuilder(
            dataset_name=self.dataset_name,
            task=self.task,
            model_name=self.ehr_model_name,
            mode=self.mode
        )

    def _create_placeholder_context_builder(self):
        """Create a placeholder context builder."""
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
        Load trial background information.

        Loads from corpus/ctgov/{dataset_name}/bg.json.
        """
        try:
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
                self.trial_bg = ""
                self.ctrl_grp = ""
                self.expr_grp = ""
        except Exception as e:
            print(f"Warning: Failed to load trial background: {e}")
            self.trial_bg = ""
            self.ctrl_grp = ""
            self.expr_grp = ""

        self.group = self.config.get("group", "experimental")

    def get_role_specific_prompts(self) -> Tuple[str, str, str, str]:
        """
        Return role-specific prompts based on clinician index.

        Returns:
            (system_prompt, user_prompt_template, collaboration_system, collaboration_user_template)
        """
        try:
            from utils.prompts.template_enhanced_diversity import (
                CLINICIAN_ROLES,
                CLINICIAN_SYSTEM_PROMPTS,
                CLINICIAN_USER_PROMPTS,
                CLINICIAN_COLLABORATION_SYSTEM_PROMPTS,
                CLINICIAN_COLLABORATION_USER_PROMPTS
            )

            idx = self.clinician_index
            system_prompt = CLINICIAN_SYSTEM_PROMPTS.get(idx, "")
            user_prompt = CLINICIAN_USER_PROMPTS.get(idx, "")
            collab_system = CLINICIAN_COLLABORATION_SYSTEM_PROMPTS.get(idx, "")
            collab_user = CLINICIAN_COLLABORATION_USER_PROMPTS.get(idx, "")

        except ImportError:
            system_prompt = self._get_default_system_prompt()
            user_prompt = self._get_default_user_prompt()
            collab_system = self._get_default_collaboration_system_prompt()
            collab_user = self._get_default_collaboration_user_prompt()

        return system_prompt, user_prompt, collab_system, collab_user

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        specialties = ["emergency_medicine", "oncology", "general_medicine"]
        specialty = specialties[self.clinician_index] if self.clinician_index < len(specialties) else "general_medicine"

        return f"""You are a medical expert specializing in {specialty.replace('_', ' ')}.
Your task is to analyze patient data and provide professional medical assessment.
Provide your analysis in JSON format with the following fields:
- analysis: Your detailed clinical analysis
- logit: Survival probability (0-1)
- conf_logit: Your confidence in this prediction (0-1)"""

    def _get_default_user_prompt(self) -> str:
        """Get default user prompt."""
        return "Please analyze the following patient data: {context}"

    def _get_default_collaboration_system_prompt(self) -> str:
        """Get default collaboration system prompt."""
        return "You are participating in a medical consultation. Review the opinions and provide your assessment."

    def _get_default_collaboration_user_prompt(self) -> str:
        """Get default collaboration user prompt."""
        return "Review the following opinions and provide your response: {opinions}"

    def _build_patient_profile_text(self, context, patient_id: str) -> str:
        """
        Build patient feature text for Memory retrieval.

        Args:
            context: Patient context (string or dict)
            patient_id: Patient ID

        Returns:
            Patient feature description text
        """
        group = self.config.get("group", "unknown")
        dataset = self.dataset_name

        if isinstance(context, dict):
            features = []
            for key in ["main_complaint", "diagnosis", "treatment", "age", "gender"]:
                if key in context and context[key]:
                    features.append(f"{key}={context[key]}")
            feature_str = ', '.join(features)
        else:
            feature_str = str(context)[:200]

        return f"Patient {patient_id} ({dataset}/{group}): {feature_str}"

    def retrieve(self, question: str, k: int = 1) -> str:
        """
        Retrieve relevant medical documents.

        Args:
            question: Query string
            k: Number of documents to retrieve

        Returns:
            Retrieved document text
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

    def analysis(self, patient_index: int, patient_id: str) -> Tuple[Dict, str, List, int, int]:
        """
        Perform clinical analysis (Phase 1).

        Generates an initial clinical review based on patient data and
        retrieved medical knowledge.

        Args:
            patient_index: Patient index
            patient_id: Patient ID

        Returns:
            (analysis_result_dict, medical_context, messages, prompt_tokens, completion_tokens)
        """
        basic_context, subcontext, healthcare_context = self.context_builder.build(
            patient_index, patient_id
        )

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

        doc_context = self.retrieve(self.trial_bg + "\n" + group_info + '\n' + subcontext)

        context = self.trial_bg + "\n" + group_info + '\n' + doc_context

        self.patient_info = healthcare_context

        system_prompt, user_prompt_template, _, _ = self.get_role_specific_prompts()

        if hasattr(user_prompt_template, 'render'):
            user_prompt = user_prompt_template.render(context=context, hcontext=healthcare_context)
        else:
            user_prompt = user_prompt_template.format(context=context, hcontext=healthcare_context)

        if self.memory_injector is not None:
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

        ans, prompt_token, completion_token = self.invoke(messages=messages)

        self.latest_analysis = ans.get("analysis", "")
        self.initial_analysis = ans.get("analysis", "")

        return ans, healthcare_context, messages, prompt_token, completion_token

    def collaboration(self, supervisor_opinion: str, supervisor_report: str, last_review: Dict) -> Tuple[Dict, List, int, int]:
        """
        Participate in collaborative discussion (Phase 2).

        Responds to the Supervisor's opinion with professional assessment.

        Args:
            supervisor_opinion: Supervisor's opinion
            supervisor_report: Supervisor's report
            last_review: Previous round analysis

        Returns:
            (response_dict, messages, prompt_tokens, completion_tokens)
        """
        context = self.trial_bg
        latest_analysis = f"In the last round, you thought the survival probability of the patient was: {last_review.get('logit', 0)}, with confidence {last_review.get('conf_logit', 0)}.\nYour detailed analysis of the patient is: {self.latest_analysis}"

        _, _, collaboration_system, collaboration_user_template = self.get_role_specific_prompts()

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

        evidences = ans.get("evidences", [])
        evidences_str = "\n".join(evidences) if isinstance(evidences, list) else str(evidences)
        self.latest_analysis = ans.get("reason", "") + evidences_str

        return ans, messages, prompt_token, completion_token

    def get_latest_analysis(self, sub_save_dir: str, idx: int) -> Dict:
        """
        Get the latest analysis result.

        Loads the latest analysis from saved files.

        Args:
            sub_save_dir: Sub save directory
            idx: Clinician index

        Returns:
            Latest analysis result dict
        """
        self.latest_reason = None
        self.latest_logit = None
        self.latest_conf = None

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
        Call LLM API.

        Args:
            messages: Message list

        Returns:
            (response_dict, prompt_tokens, completion_tokens)
        """
        start = time.time()

        try:
            config = self.get_generation_config()

            if 'gemini' not in self.llm.lower():
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
                response = self.client.models.generate_content(
                    model=self.llm,
                    contents=[messages[0]['content'], messages[1]['content']],
                )
                content = response.text
                prompt_token = response.usage_metadata.prompt_token_count
                completion_token = response.usage_metadata.candidates_token_count

            from .json_parser import extract_and_parse_json
            ans = extract_and_parse_json(content)

        except Exception as e:
            raise e

        end = time.time()
        print(f"ClinicianAgent-{self.clinician_index} (temp={config['temperature']:.2f}) time cost: {end - start:.2f}s.")

        return ans, prompt_token, completion_token
