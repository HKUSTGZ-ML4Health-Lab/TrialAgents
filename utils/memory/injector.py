"""
TrialAgents Memory Bank - Prompt注入

将检索到的相似案例注入到Agent的prompt中
"""

from typing import List, Optional, Dict, Any
from .engine import MemoryEngine, SearchResult
from .schema import ErrorCase, SuccessCase


class MemoryInjector:
    """
    记忆注入器

    负责将相似案例检索结果注入到ClinicianAgent和SupervisorAgent的prompt中
    """

    # 注入模板
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
        初始化注入器

        Args:
            memory_engine: MemoryEngine实例
        """
        self.engine = memory_engine

    def build_error_warning(self, error_results: List[SearchResult]) -> str:
        """
        构建错误警示文本

        Args:
            error_results: 检索到的相似错误

        Returns:
            格式化的错误警示文本
        """
        if not error_results:
            return ""

        warnings = []
        for idx, result in enumerate(error_results[:3], 1):  # 最多3个
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
        构建成功案例参考文本

        Args:
            success_results: 检索到的相似成功案例

        Returns:
            格式化的成功案例参考文本
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
        为ClinicianAgent构建带有记忆上下文的prompt

        Args:
            base_prompt: 原始prompt
            patient_profile_text: 患者特征文本（用于检索）
            error_types: 关注的错误类型列表
            k_errors: 检索的错误案例数量

        Returns:
            注入记忆后的prompt
        """
        # 检索相似错误
        error_results = self.engine.search_similar_errors(
            patient_profile_text=patient_profile_text,
            k=k_errors,
            error_types=error_types,
        )

        # 构建错误警示
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
        为SupervisorAgent构建带有记忆上下文的prompt

        Args:
            base_prompt: 原始prompt
            patient_profile_text: 患者特征文本（用于检索）
            k_errors: 检索的错误案例数量
            k_successes: 检索的成功案例数量

        Returns:
            注入记忆后的prompt
        """
        # 检索相似错误和成功案例
        error_results = self.engine.search_similar_errors(
            patient_profile_text=patient_profile_text,
            k=k_errors,
        )

        success_results = self.engine.search_similar_successes(
            patient_profile_text=patient_profile_text,
            k=k_successes,
            min_confidence=0.7,
        )

        # 构建记忆文本
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
        获取所有相似案例（用于调试和分析）

        Args:
            patient_profile_text: 患者特征文本
            k: 检索数量

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
    Prompt增强器

    提供额外的prompt增强功能，如Chain-of-Thought结构化
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
        增强prompt为Chain-of-Thought格式

        Args:
            base_prompt: 原始prompt
            patient_profile_text: 患者特征文本（可选，用于记忆注入）

        Returns:
            增强后的prompt
        """
        enhanced = base_prompt

        # 如果有记忆注入器，添加记忆上下文
        if self.memory_injector and patient_profile_text:
            enhanced = self.memory_injector.inject_for_clinician(
                base_prompt=enhanced,
                patient_profile_text=patient_profile_text,
                k_errors=3,
            )

        return enhanced

    def get_cot_system_prompt(self) -> str:
        """获取Chain-of-Thought系统prompt"""
        return self.COT_SYSTEM_PROMPT
