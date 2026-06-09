"""
TrialAgents Memory Bank - Prompt Injection

Injects retrieved similar cases into the Agent's prompt
"""

from typing import List, Optional, Dict, Any
from .engine import MemoryEngine, SearchResult
from .schema import ErrorCase, SuccessCase


class MemoryInjector:
    """
    Memory injector

    Responsible for injecting similar case search results into
    ClinicianAgent and SupervisorAgent prompts
    """

    # Injection templates
    ERROR_WARNING_TEMPLATE = """
## ⚠️ Similar Past Errors (Learning from Mistakes)

Based on previous similar cases, be aware of the following potential errors:

{error_warnings}

Please analyze this patient carefully, avoiding these common pitfalls.
"""

    ERROR_ITEM_TEMPLATE = """
### Case {idx}: {error_type}
- **Patient Profile**: {patient_text}
- **What went wrong**: {root_cause}
- **Improvement hint**: {improvement_hint}
"""

    SUCCESS_TEMPLATE = """
## ✅ Similar Past Successes (Effective Patterns)

Reference these successfully analyzed cases for effective analysis patterns:

{success_cases}
"""

    SUCCESS_ITEM_TEMPLATE = """
### Case {idx}: (Confidence: {confidence:.2f})
- **Patient Profile**: {patient_text}
- **Effective reasoning**: {reasoning_chain[:200]}...
- **Key triggers**: {key_triggers}
"""

    MEMORY_CONTEXT_HEADER = """
## 📚 Memory Context (Auto-Retrieved)

This section contains relevant past cases retrieved from memory to assist your analysis.
"""

    def __init__(self, memory_engine: MemoryEngine):
        """
        Initialize injector

        Args:
            memory_engine: MemoryEngine instance
        """
        self.engine = memory_engine

    def build_error_warning(self, error_results: List[SearchResult]) -> str:
        """
        Build error warning text

        Args:
            error_results: Retrieved similar errors

        Returns:
            Formatted error warning text
        """
        if not error_results:
            return ""

        warnings = []
        for idx, result in enumerate(error_results[:3], 1):  # Maximum of 3
            if result.case_type != "error":
                continue

            error_case = result.data
            warning = self.ERROR_ITEM_TEMPLATE.format(
                idx=idx,
                error_type=error_case.error_type.value,
                patient_text=error_case.patient_profile.to_text(),
                root_cause=error_case.root_cause,
                improvement_hint=error_case.improvement_hint,
            )
            warnings.append(warning)

        if warnings:
            return self.ERROR_WARNING_TEMPLATE.format(
                error_warnings="\n".join(warnings)
            )
        return ""

    def build_success_reference(self, success_results: List[SearchResult]) -> str:
        """
        Build success case reference text

        Args:
            success_results: Retrieved similar success cases

        Returns:
            Formatted success case reference text
        """
        if not success_results:
            return ""

        cases = []
        for idx, result in enumerate(success_results[:3], 1):
            if result.case_type != "success":
                continue

            success_case = result.data
            case = self.SUCCESS_ITEM_TEMPLATE.format(
                idx=idx,
                confidence=success_case.confidence,
                patient_text=success_case.patient_profile.to_text(),
                reasoning_chain=success_case.reasoning_chain,
                key_triggers=", ".join(success_case.key_triggers[:3]),
            )
            cases.append(case)

        if cases:
            return self.SUCCESS_TEMPLATE.format(
                success_cases="\n".join(cases)
            )
        return ""

    def inject_for_clinician(
        self,
        base_prompt: str,
        patient_profile_text: str,
        error_types: List[str] = None,
        k_errors: int = 3,
    ) -> str:
        """
        Build prompt with memory context for ClinicianAgent

        Args:
            base_prompt: Original prompt
            patient_profile_text: Patient feature text (for search)
            error_types: List of error types to focus on
            k_errors: Number of error cases to retrieve

        Returns:
            Prompt with injected memory
        """
        # Search for similar errors
        error_results = self.engine.search_similar_errors(
            patient_profile_text=patient_profile_text,
            k=k_errors,
            error_types=error_types,
        )

        # Build error warning
        error_warning = self.build_error_warning(error_results)

        if error_warning:
            return base_prompt + self.MEMORY_CONTEXT_HEADER + error_warning

        return base_prompt

    def inject_for_supervisor(
        self,
        base_prompt: str,
        patient_profile_text: str,
        k_errors: int = 2,
        k_successes: int = 2,
    ) -> str:
        """
        Build prompt with memory context for SupervisorAgent

        Args:
            base_prompt: Original prompt
            patient_profile_text: Patient feature text (for search)
            k_errors: Number of error cases to retrieve
            k_successes: Number of success cases to retrieve

        Returns:
            Prompt with injected memory
        """
        # Search for similar errors and success cases
        error_results = self.engine.search_similar_errors(
            patient_profile_text=patient_profile_text,
            k=k_errors,
        )

        success_results = self.engine.search_similar_successes(
            patient_profile_text=patient_profile_text,
            k=k_successes,
            min_confidence=0.7,
        )

        # Build memory text
        memory_parts = []

        if error_results:
            memory_parts.append("### Past Errors to Avoid:")
            memory_parts.append(self.build_error_warning(error_results))

        if success_results:
            memory_parts.append("\n### Past Successes to Reference:")
            memory_parts.append(self.build_success_reference(success_results))

        if memory_parts:
            return base_prompt + self.MEMORY_CONTEXT_HEADER + "\n".join(memory_parts)

        return base_prompt

    def get_similar_cases_for_review(
        self,
        patient_profile_text: str,
        k: int = 5,
    ) -> Dict[str, List[SearchResult]]:
        """
        Get all similar cases (for debugging and analysis)

        Args:
            patient_profile_text: Patient feature text
            k: Number of results to retrieve

        Returns:
            {"errors": [...], "successes": [...]}
        """
        error_results = self.engine.search_similar_errors(
            patient_profile_text=patient_profile_text,
            k=k,
        )

        success_results = self.engine.search_similar_successes(
            patient_profile_text=patient_profile_text,
            k=k,
        )

        return {
            "errors": error_results,
            "successes": success_results,
        }


class PromptEnhancer:
    """
    Prompt enhancer

    Provides additional prompt enhancement features, such as Chain-of-Thought structuring
    """

    COT_SYSTEM_PROMPT = """You are a medical expert analyzing patient data.
Follow the Chain-of-Thought structure strictly:

1. **Observation**: State the key patient features observed
2. **Analysis**: Provide detailed clinical analysis
3. **Reasoning**: Explain the causal reasoning
4. **Risk Assessment**: Evaluate the risk level
5. **Confidence**: State your confidence level and reasoning

Output your analysis in JSON format with these fields:
- observation: (str)
- analysis: (str)
- reasoning: (str)
- risk_assessment: (str)
- confidence: (float, 0-1)
- logit: (float, 0-1)
- conf_logit: (float, 0-1)
- evidences: (list of str)
"""

    def __init__(self, memory_injector: MemoryInjector = None):
        self.memory_injector = memory_injector

    def enhance_for_cot(self, base_prompt: str, patient_profile_text: str = None) -> str:
        """
        Enhance prompt to Chain-of-Thought format

        Args:
            base_prompt: Original prompt
            patient_profile_text: Patient feature text (optional, for memory injection)

        Returns:
            Enhanced prompt
        """
        enhanced = base_prompt

        # If memory injector is available, add memory context
        if self.memory_injector and patient_profile_text:
            enhanced = self.memory_injector.inject_for_clinician(
                base_prompt=enhanced,
                patient_profile_text=patient_profile_text,
                k_errors=3,
            )

        return enhanced

    def get_cot_system_prompt(self) -> str:
        """Get Chain-of-Thought system prompt"""
        return self.COT_SYSTEM_PROMPT
