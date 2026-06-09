"""
TrialAgents Memory Bank - Auto Updater

Responsible for automatically registering prediction errors into the Memory Bank
Includes root cause analysis and improvement hint generation
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
    Memory auto updater

    Responsibilities:
    1. Prediction result evaluation (whether it was an error)
    2. Automatic error case registration
    3. Root cause analysis (using LLM)
    4. Improvement hint generation
    5. Success case registration
    """

    def __init__(
        self,
        store: MemoryStore,
        engine: MemoryEngine,
        llm_client=None,  # Optional: for root cause analysis
    ):
        """
        Initialize updater

        Args:
            store: MemoryStore instance
            engine: MemoryEngine instance
            llm_client: Optional LLM client for root cause analysis
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
        Register prediction result, automatically determine whether it was an error and register

        Args:
            case_id: Case ID
            dataset: Dataset name
            patient_profile: Patient profile
            clinician_index: Clinician index
            clinician_analysis: Clinician analysis text
            prediction: Prediction value (0-1)
            ground_truth: Ground truth label (0 or 1)
            clinician_confidence: Clinician confidence

        Returns:
            (is_error, error_case_id or success_case_id)
        """
        # Determine if prediction is wrong
        predicted_class = 1 if prediction >= 0.5 else 0
        is_error = predicted_class != ground_truth

        # Determine error type
        error_type = self._classify_error(prediction, ground_truth, clinician_confidence)

        if is_error:
            # Generate root cause analysis
            root_cause, improvement_hint = self._analyze_error(
                clinician_analysis=clinician_analysis,
                prediction=prediction,
                ground_truth=ground_truth,
                error_type=error_type,
            )

            # Create error case
            error_case = ErrorCase(
                case_id=case_id,
                dataset=dataset,
                patient_profile=patient_profile,
                clinician_analysis=clinician_analysis,
                error_type=error_type,
                root_cause=root_cause,
                correction="",  # Can be supplemented later
                improvement_hint=improvement_hint,
                timestamp=datetime.now(),
                frequency=1,
                clinician_index=clinician_index,
                prediction=prediction,
                ground_truth=ground_truth,
            )

            # Store
            stored_id = self.store.add_error(error_case)

            # Update index
            self.engine.add_error_case(error_case)

            return True, stored_id

        else:
            # Create success case
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

            # Store
            stored_id = self.store.add_success(success_case)

            # Update index
            self.engine.add_success_case(success_case)

            return False, stored_id

    def _classify_error(
        self,
        prediction: float,
        ground_truth: int,
        confidence: float,
    ) -> ErrorType:
        """
        Classify error type

        Args:
            prediction: Prediction value
            ground_truth: Ground truth label
            confidence: Confidence

        Returns:
            ErrorType
        """
        predicted_class = 1 if prediction >= 0.5 else 0

        if predicted_class == 0 and ground_truth == 1:
            # False negative (underestimated risk)
            if confidence < 0.5:
                return ErrorType.UNDERESTIMATION
            else:
                return ErrorType.WRONG_BASIS
        elif predicted_class == 1 and ground_truth == 0:
            # False positive (overestimated risk)
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
        Analyze error root cause and generate improvement hint

        Args:
            clinician_analysis: Clinician analysis text
            prediction: Prediction value
            ground_truth: Ground truth label
            error_type: Error type

        Returns:
            (root_cause, improvement_hint)
        """
        # If LLM client is available, use LLM analysis
        if self.llm_client is not None:
            try:
                return self._llm_analyze_error(
                    clinician_analysis, prediction, ground_truth, error_type
                )
            except Exception as e:
                print(f"Warning: LLM error analysis failed: {e}")

        # Fallback: simple rule-based analysis
        return self._rule_based_analysis(prediction, ground_truth, error_type)

    def _llm_analyze_error(
        self,
        clinician_analysis: str,
        prediction: float,
        ground_truth: int,
        error_type: ErrorType,
    ) -> Tuple[str, str]:
        """
        Use LLM to analyze error root cause

        Args:
            clinician_analysis: Clinician analysis
            prediction: Prediction value
            ground_truth: Ground truth label
            error_type: Error type

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

            # Parse JSON
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
        Simple rule-based error analysis

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
        Extract key trigger words from analysis

        Args:
            analysis: Analysis text

        Returns:
            List of key trigger words
        """
        # Simple keyword extraction
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

        # Return at most 5
        return triggers[:5]

    def update_from_batch_results(
        self,
        results_file: str,
        clinician_analysis_file: str,
        dataset: str,
    ) -> Dict[str, Any]:
        """
        Batch update memory from batch result files

        Args:
            results_file: supervisor_logits_json.json file path
            clinician_analysis_file: clinician{idx}_review.json file path
            dataset: Dataset name

        Returns:
            Update statistics
        """
        stats = {
            "total": 0,
            "errors": 0,
            "successes": 0,
            "failed": 0,
        }

        # Read prediction results
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error reading results file: {e}")
            return stats

        # Read clinician analyses
        clinician_analyses = {}
        for idx in range(3):  # Assuming 3 clinicians
            doc_file = clinician_analysis_file.replace("{idx}", str(idx + 1))
            if os.path.exists(doc_file):
                try:
                    with open(doc_file, "r") as f:
                        clinician_analyses[idx] = json.load(f)
                except Exception:
                    pass

        # Iterate over each patient
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

                # Get clinician analysis
                clinician_analysis = ""
                if clinician_analyses:
                    first_analysis = next(iter(clinician_analyses.values()), {})
                    clinician_analysis = first_analysis.get("analysis", "")

                # Register result
                is_error, _ = self.register_prediction_result(
                    case_id=case_id,
                    dataset=dataset,
                    patient_profile=patient_profile,
                    clinician_index=0,  # Default to first clinician
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

        # Save index
        self.engine.save_index()

        return stats


def create_memory_updater(config: Dict = None) -> MemoryUpdater:
    """
    Factory function: create MemoryUpdater

    Args:
        config: Configuration dictionary

    Returns:
        MemoryUpdater instance
    """
    from .store import MemoryStore
    from .engine import MemoryEngine

    # Create store and engine
    store = MemoryStore()
    engine = MemoryEngine(store=store)

    # Optional: create LLM client
    llm_client = None
    if config and config.get("use_llm_analysis", False):
        try:
            from utils.llm.client import LLMClient
            llm_client = LLMClient(model=config.get("llm", "gpt-4o-mini"))
        except Exception as e:
            print(f"Warning: Failed to create LLM client: {e}")

    return MemoryUpdater(store=store, engine=engine, llm_client=llm_client)
