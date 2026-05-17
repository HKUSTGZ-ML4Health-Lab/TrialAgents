"""
TrialAgents Memory Bank - 自动更新器

负责在预测错误时自动登记到Memory Bank
包括根因分析和改进提示生成
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import asdict

from .store import MemoryStore
from .engine import MemoryEngine
from .schema import ErrorCase, SuccessCase, PatientProfile, ErrorType


class MemoryUpdater:
    """
    记忆自动更新器

    负责：
    1. 预测结果评估（是否错误）
    2. 错误案例自动登记
    3. 根因分析（使用LLM）
    4. 改进提示生成
    5. 成功案例登记
    """

    def __init__(
        self,
        store: MemoryStore,
        engine: MemoryEngine,
        llm_client=None,  # 可选：用于根因分析
    ):
        """
        初始化更新器

        Args:
            store: MemoryStore实例
            engine: MemoryEngine实例
            llm_client: 可选的LLM客户端，用于根因分析
        """
        self.store = store
        self.engine = engine
        self.llm_client = llm_client

    def register_prediction_result(
        self,
        case_id: str,
        dataset: str,
        patient_profile: PatientProfile,
        clinician_index: int,
        clinician_analysis: str,
        prediction: float,
        ground_truth: int,
        clinician_confidence: float = 0.5,
    ) -> Tuple[bool, Optional[str]]:
        """
        登记预测结果，自动判断是否错误并登记

        Args:
            case_id: 案例ID
            dataset: 数据集名称
            patient_profile: 患者特征
            clinician_index: clinician索引
            clinician_analysis: clinician分析文本
            prediction: 预测值 (0-1)
            ground_truth: 真实标签 (0或1)
            clinician_confidence: clinician置信度

        Returns:
            (is_error, error_case_id 或 success_case_id)
        """
        # 判断是否错误
        predicted_class = 1 if prediction >= 0.5 else 0
        is_error = predicted_class != ground_truth

        # 判断错误类型
        error_type = self._classify_error(prediction, ground_truth, clinician_confidence)

        if is_error:
            # 生成根因分析
            root_cause, improvement_hint = self._analyze_error(
                clinician_analysis=clinician_analysis,
                prediction=prediction,
                ground_truth=ground_truth,
                error_type=error_type,
            )

            # 创建错误案例
            error_case = ErrorCase(
                case_id=case_id,
                dataset=dataset,
                patient_profile=patient_profile,
                clinician_analysis=clinician_analysis,
                error_type=error_type,
                root_cause=root_cause,
                correction="",  # 稍后可以补充
                improvement_hint=improvement_hint,
                timestamp=datetime.now(),
                frequency=1,
                clinician_index=clinician_index,
                prediction=prediction,
                ground_truth=ground_truth,
            )

            # 存储
            stored_id = self.store.add_error(error_case)

            # 更新索引
            self.engine.add_error_case(error_case)

            return True, stored_id

        else:
            # 创建成功案例
            success_case = SuccessCase(
                case_id=case_id,
                dataset=dataset,
                patient_profile=patient_profile,
                reasoning_chain=clinician_analysis,
                outcome=f"Correct prediction: {predicted_class} (ground_truth={ground_truth})",
                confidence=clinician_confidence,
                key_triggers=self._extract_key_triggers(clinician_analysis),
                timestamp=datetime.now(),
                clinician_index=clinician_index,
            )

            # 存储
            stored_id = self.store.add_success(success_case)

            # 更新索引
            self.engine.add_success_case(success_case)

            return False, stored_id

    def _classify_error(
        self,
        prediction: float,
        ground_truth: int,
        confidence: float,
    ) -> ErrorType:
        """
        分类错误类型

        Args:
            prediction: 预测值
            ground_truth: 真实标签
            confidence: 置信度

        Returns:
            ErrorType
        """
        predicted_class = 1 if prediction >= 0.5 else 0

        if predicted_class == 0 and ground_truth == 1:
            # 漏报（低估风险）
            if confidence < 0.5:
                return ErrorType.UNDERESTIMATION
            else:
                return ErrorType.WRONG_BASIS
        elif predicted_class == 1 and ground_truth == 0:
            # 误报（高估风险）
            if confidence > 0.5:
                return ErrorType.OVERESTIMATION
            else:
                return ErrorType.WRONG_BASIS
        else:
            return ErrorType.UNKNOWN

    def _analyze_error(
        self,
        clinician_analysis: str,
        prediction: float,
        ground_truth: int,
        error_type: ErrorType,
    ) -> Tuple[str, str]:
        """
        分析错误根因并生成改进提示

        Args:
            clinician_analysis: clinician分析文本
            prediction: 预测值
            ground_truth: 真实标签
            error_type: 错误类型

        Returns:
            (root_cause, improvement_hint)
        """
        # 如果有LLM客户端，使用LLM分析
        if self.llm_client is not None:
            try:
                return self._llm_analyze_error(
                    clinician_analysis, prediction, ground_truth, error_type
                )
            except Exception as e:
                print(f"Warning: LLM error analysis failed: {e}")

        # Fallback: 基于规则的简单分析
        return self._rule_based_analysis(prediction, ground_truth, error_type)

    def _llm_analyze_error(
        self,
        clinician_analysis: str,
        prediction: float,
        ground_truth: int,
        error_type: ErrorType,
    ) -> Tuple[str, str]:
        """
        使用LLM分析错误根因

        Args:
            clinician_analysis: clinician分析
            prediction: 预测值
            ground_truth: 真实标签
            error_type: 错误类型

        Returns:
            (root_cause, improvement_hint)
        """
        system_prompt = """You are a medical error analyst. Given a clinician's analysis that led to an incorrect prediction, identify:
1. The root cause of the error (1-2 sentences)
2. Specific improvement hints for future analysis (1-2 sentences)

Output in JSON format:
{
    "root_cause": "...",
    "improvement_hint": "..."
}
"""

        user_prompt = f"""Clinician's analysis:
{clinician_analysis}

Prediction: {prediction} (Predicted class: {'survived' if prediction < 0.5 else 'died'})
Ground truth: {ground_truth} ({'survived' if ground_truth == 0 else 'died'})
Error type: {error_type.value}

Please analyze the error and provide root cause and improvement hint."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_client.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content

            # 解析JSON
            from src.agents.json_parser import extract_and_parse_json
            result = extract_and_parse_json(content)

            return result.get("root_cause", "Unknown root cause"), result.get("improvement_hint", "Review patient data more carefully")

        except Exception as e:
            print(f"Warning: LLM analysis failed: {e}")
            return self._rule_based_analysis(prediction, ground_truth, error_type)

    def _rule_based_analysis(
        self,
        prediction: float,
        ground_truth: int,
        error_type: ErrorType,
    ) -> Tuple[str, str]:
        """
        基于规则的简单错误分析

        Returns:
            (root_cause, improvement_hint)
        """
        if error_type == ErrorType.UNDERESTIMATION:
            return (
                "The model underestimated the patient's risk, potentially missing critical factors.",
                "Consider additional risk factors and Comorbidities in the analysis."
            )
        elif error_type == ErrorType.OVERESTIMATION:
            return (
                "The model overestimated the patient's risk, potentially due to over-weighting certain factors.",
                "Review whether all risk factors are truly independent and consider base rates."
            )
        elif error_type == ErrorType.WRONG_BASIS:
            return (
                "The analysis may have relied on incorrect or incomplete medical evidence.",
                "Cross-reference with more authoritative medical literature and consider alternative diagnoses."
            )
        else:
            return (
                "The prediction did not match the ground truth.",
                "Review the patient's complete medical history and ensure all relevant factors are considered."
            )

    def _extract_key_triggers(self, analysis: str) -> List[str]:
        """
        从分析中提取关键触发词

        Args:
            analysis: 分析文本

        Returns:
            关键触发词列表
        """
        # 简单的关键词提取
        trigger_keywords = [
            "high risk", "low risk", "critical", "stable",
            "comorbidity", "age", "prognosis", "treatment",
            "response", "tolerance", "adverse", "benefit",
        ]

        triggers = []
        analysis_lower = analysis.lower()

        for keyword in trigger_keywords:
            if keyword in analysis_lower:
                triggers.append(keyword)

        # 返回最多5个
        return triggers[:5]

    def update_from_batch_results(
        self,
        results_file: str,
        clinician_analysis_file: str,
        dataset: str,
    ) -> Dict[str, Any]:
        """
        从批量结果文件批量更新记忆

        Args:
            results_file: supervisor_logits_json.json 文件路径
            clinician_analysis_file: clinician{idx}_review.json 文件路径
            dataset: 数据集名称

        Returns:
            更新统计
        """
        stats = {
            "total": 0,
            "errors": 0,
            "successes": 0,
            "failed": 0,
        }

        # 读取预测结果
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error reading results file: {e}")
            return stats

        # 读取clinician分析
        clinician_analyses = {}
        for idx in range(3):  # 假设3个医生
            doc_file = clinician_analysis_file.replace("{idx}", str(idx + 1))
            if os.path.exists(doc_file):
                try:
                    with open(doc_file, "r") as f:
                        clinician_analyses[idx] = json.load(f)
                except Exception:
                    pass

        # 遍历每个患者
        for case_id, result in results.items():
            if case_id.startswith("total") or case_id == "processed_count":
                continue

            stats["total"] += 1

            try:
                prediction = result.get("final_logit", 0.5)
                ground_truth = result.get("ground_truth")
                patient_profile_data = result.get("patient_profile", {})

                if ground_truth is None:
                    continue

                patient_profile = PatientProfile(
                    patient_id=case_id,
                    dataset=dataset,
                    group=patient_profile_data.get("group", "unknown"),
                    key_features=patient_profile_data.get("key_features", {}),
                    prediction=prediction,
                    ground_truth=ground_truth,
                )

                # 获取clinician分析
                clinician_analysis = ""
                if clinician_analyses:
                    first_analysis = next(iter(clinician_analyses.values()), {})
                    clinician_analysis = first_analysis.get("analysis", "")

                # 登记结果
                is_error, _ = self.register_prediction_result(
                    case_id=case_id,
                    dataset=dataset,
                    patient_profile=patient_profile,
                    clinician_index=0,  # 默认第一个医生
                    clinician_analysis=clinician_analysis,
                    prediction=prediction,
                    ground_truth=ground_truth,
                )

                if is_error:
                    stats["errors"] += 1
                else:
                    stats["successes"] += 1

            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
                stats["failed"] += 1

        # 保存索引
        self.engine.save_index()

        return stats


def create_memory_updater(config: Dict = None) -> MemoryUpdater:
    """
    工厂函数：创建MemoryUpdater

    Args:
        config: 配置字典

    Returns:
        MemoryUpdater实例
    """
    from .store import MemoryStore
    from .engine import MemoryEngine

    # 创建存储和引擎
    store = MemoryStore()
    engine = MemoryEngine(store=store)

    # 可选：创建LLM客户端
    llm_client = None
    if config and config.get("use_llm_analysis", False):
        try:
            from utils.llm.client import LLMClient
            llm_client = LLMClient(model=config.get("llm", "gpt-4o-mini"))
        except Exception as e:
            print(f"Warning: Failed to create LLM client: {e}")

    return MemoryUpdater(store=store, engine=engine, llm_client=llm_client)
