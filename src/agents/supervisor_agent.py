"""
TrialAgents SupervisorAgent (Trialist + Supervisor)

Migrated and refactored from m-agents/utils/framework.py
Integrates Trialist and Supervisor Agent functionality:
- Trialist: Synthesizes multi-clinician reviews into dynamic reports
- Supervisor: Coordinates discussion and generates final reports
"""

import time
from typing import Dict, List, Tuple
from tenacity import retry, wait_random_exponential, stop_after_attempt

from .base_agent import Agent


class SupervisorAgent(Agent):
    """
    Supervisor Agent / Meta Agent

    Two core functions:
    1. Trialist: Synthesizes multiple ClinicianAgent reviews into dynamic summary reports
    2. Supervisor: Coordinates discussion, integrates all perspectives, generates final report

    Attributes:
        current_report: Current report content
        basic_info: Patient basic information
    """

    def __init__(
        self,
        role: str = "supervisor",
        llm: str = None,
        memory_injector=None,
    ) -> None:
        """
        Initialize SupervisorAgent.

        Args:
            role: Role name
            llm: LLM model name
            memory_injector: MemoryInjector instance
        """
        super().__init__(role, llm)

        self.current_report = ""
        self.basic_info = None
        self.memory_injector = memory_injector

        self._init_prompts()

    def _init_prompts(self):
        """Initialize prompt templates."""
        try:
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
            self._init_default_prompts()

    def _init_default_prompts(self):
        """Initialize default prompts."""
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
        Set patient basic info.

        Args:
            patient_info: Patient information text
        """
        self.basic_info = patient_info

    def _render_template(self, template, **kwargs) -> str:
        """
        Render template, compatible with liquid.Template.render() and str.format().
        """
        if hasattr(template, 'render'):
            return template.render(**kwargs)
        else:
            return template.format(**kwargs)

    def generate_clinicians_prompt(self, clinician_responses: List[Dict], is_initial: bool = False) -> str:
        """
        Generate prompt text from clinician opinions.

        Args:
            clinician_responses: List of clinician response dicts
            is_initial: Whether this is the initial review

        Returns:
            Formatted clinician opinion text
        """
        responses = ""
        for idx, res in enumerate(clinician_responses):
            if is_initial:
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
        Extract action response.

        Args:
            response: LLM response dict

        Returns:
            1 to continue discussion, 0 to finalize
        """
        action = response.get('action', '').lower()
        if "yes" in action or "continue" in action or "1" in action:
            return 1
        elif "no" in action or "finalize" in action or "0" in action:
            return 0
        else:
            print(f"Warning: Invalid action response: {action}. Defaulting to continue.")
            return 1

    def next_action(self, clinician_responses: List[Dict]) -> Tuple[int, Dict, List, int, int]:
        """
        Decide next action.

        Analyzes clinician responses and decides whether to continue discussion.

        Args:
            clinician_responses: Clinician response list, each containing:
                - answer: agree/disagree
                - confidence: 0-1
                - reason: rationale
                - evidences: evidence items

        Returns:
            (action, response_dict, messages, prompt_tokens, completion_tokens)
            action: 1 continue loop, 0 end loop
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
        Generate/update report (Trialist function).

        Synthesizes clinician opinions into a dynamic summary report.

        Args:
            clinician_responses: Clinician response list
            is_initial: Whether this is the initial review

        Returns:
            (response_dict, messages, prompt_tokens, completion_tokens)
        """
        clinicians_answer_prompt = self.generate_clinicians_prompt(clinician_responses, is_initial)

        if is_initial:
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
        Generate initial clinician opinion prompt (for revise_logits).

        Args:
            clinician_responses: Clinician response list

        Returns:
            Formatted prompt text
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
        Revise and integrate final predictions (Supervisor function).

        Generates final report based on all clinicians' final opinions.

        Args:
            clinician_initial_responses: Clinicians' final analysis results
            patient_profile_text: Patient feature text (for Memory retrieval)

        Returns:
            (final_prediction_dict, messages, prompt_tokens, completion_tokens)
        """
        clinicians_answer_prompt = self.generate_clinicians_initial_prompt(clinician_initial_responses)
        user_prompt = self._render_template(self.revise_logits_user,
            clinician_info=clinicians_answer_prompt,
            latest_info=self.current_report
        )

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
        Call LLM API.

        SupervisorAgent uses more conservative parameter settings for
        consistency and accuracy.

        Args:
            messages: Message list

        Returns:
            (response_dict, prompt_tokens, completion_tokens)
        """
        start = time.time()

        try:
            if 'gemini' not in self.llm.lower():
                response = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    stream=False,
                    temperature=0.3,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=2048,
                    seed=None
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
        print(f"SupervisorAgent (temp=0.3) time cost: {end - start:.2f}s.")

        return ans, prompt_token, completion_token

    def get_generation_config(self) -> Dict:
        """
        Get generation config (SupervisorAgent uses conservative parameters).

        Returns:
            Generation parameters dict
        """
        return {
            "temperature": 0.3,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 2048,
            "seed": None,
        }
