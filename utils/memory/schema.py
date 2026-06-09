"""
TrialAgents Memory Bank - Data Structure Definitions

Defines the storage structures for error cases, success cases, and pattern metadata
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ErrorType(Enum):
    """Error type enumeration"""
    UNDERESTIMATION = "underestimation"       # Underestimated risk
    OVERESTIMATION = "overestimation"        # Overestimated risk
    MISDIAGNOSIS = "misdiagnosis"           # Misdiagnosis
    WRONG_BASIS = "wrong_basis"             # Wrong basis
    INCOMPLETE_ANALYSIS = "incomplete_analysis"  # Incomplete analysis
    CONTRADICTION = "contradiction"          # Logical contradiction
    UNKNOWN = "unknown"                      # Unknown type


@dataclass
class PatientProfile:
    """Patient feature summary (for similarity retrieval)"""
    patient_id: str
    dataset: str
    group: str                               # control/experimental
    key_features: Dict[str, Any]            # Key feature summary
    prediction: float                        # Prediction value
    ground_truth: Optional[int] = None      # Ground truth label

    def to_text(self) -> str:
        """Convert to retrieval text"""
        features_str = ", ".join([f"{k}={v}" for k, v in self.key_features.items()])
        return f"Patient {self.patient_id} ({self.dataset}/{self.group}): {features_str}, prediction={self.prediction}"


@dataclass
class ErrorCase:
    """Error case"""
    case_id: str
    dataset: str
    patient_profile: PatientProfile
    clinician_analysis: str                    # Erroneous analysis content
    error_type: ErrorType
    root_cause: str                         # Error root cause (LLM-generated)
    correction: str                         # Correct analysis
    improvement_hint: str                    # Improvement hint
    timestamp: datetime = field(default_factory=datetime.now)
    frequency: int = 1                      # Occurrence frequency
    clinician_index: int = 0                   # Index of the clinician who made the error
    prediction: float = 0.0                 # Erroneous prediction value
    ground_truth: int = 0                   # Ground truth label

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "patient_profile": {
                "patient_id": self.patient_profile.patient_id,
                "dataset": self.patient_profile.dataset,
                "group": self.patient_profile.group,
                "key_features": self.patient_profile.key_features,
                "prediction": self.patient_profile.prediction,
                "ground_truth": self.patient_profile.ground_truth,
            },
            "clinician_analysis": self.clinician_analysis,
            "error_type": self.error_type.value,
            "root_cause": self.root_cause,
            "correction": self.correction,
            "improvement_hint": self.improvement_hint,
            "timestamp": self.timestamp.isoformat(),
            "frequency": self.frequency,
            "clinician_index": self.clinician_index,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ErrorCase":
        """Create from dictionary"""
        patient_profile = PatientProfile(
            patient_id=data["patient_profile"]["patient_id"],
            dataset=data["patient_profile"]["dataset"],
            group=data["patient_profile"]["group"],
            key_features=data["patient_profile"]["key_features"],
            prediction=data["patient_profile"]["prediction"],
            ground_truth=data["patient_profile"].get("ground_truth"),
        )
        return cls(
            case_id=data["case_id"],
            dataset=data["dataset"],
            patient_profile=patient_profile,
            clinician_analysis=data["clinician_analysis"],
            error_type=ErrorType(data["error_type"]),
            root_cause=data["root_cause"],
            correction=data["correction"],
            improvement_hint=data["improvement_hint"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            frequency=data.get("frequency", 1),
            clinician_index=data.get("clinician_index", 0),
            prediction=data.get("prediction", 0.0),
            ground_truth=data.get("ground_truth", 0),
        )


@dataclass
class SuccessCase:
    """Success case"""
    case_id: str
    dataset: str
    patient_profile: PatientProfile
    reasoning_chain: str                   # Successful reasoning chain
    outcome: str                            # Prediction outcome description
    confidence: float                       # Confidence
    key_triggers: List[str]                 # Effective trigger points
    timestamp: datetime = field(default_factory=datetime.now)
    clinician_index: int = 0                   # Clinician index

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "patient_profile": {
                "patient_id": self.patient_profile.patient_id,
                "dataset": self.patient_profile.dataset,
                "group": self.patient_profile.group,
                "key_features": self.patient_profile.key_features,
                "prediction": self.patient_profile.prediction,
                "ground_truth": self.patient_profile.ground_truth,
            },
            "reasoning_chain": self.reasoning_chain,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "key_triggers": self.key_triggers,
            "timestamp": self.timestamp.isoformat(),
            "clinician_index": self.clinician_index,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SuccessCase":
        """Create from dictionary"""
        patient_profile = PatientProfile(
            patient_id=data["patient_profile"]["patient_id"],
            dataset=data["patient_profile"]["dataset"],
            group=data["patient_profile"]["group"],
            key_features=data["patient_profile"]["key_features"],
            prediction=data["patient_profile"]["prediction"],
            ground_truth=data["patient_profile"].get("ground_truth"),
        )
        return cls(
            case_id=data["case_id"],
            dataset=data["dataset"],
            patient_profile=patient_profile,
            reasoning_chain=data["reasoning_chain"],
            outcome=data["outcome"],
            confidence=data["confidence"],
            key_triggers=data["key_triggers"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            clinician_index=data.get("clinician_index", 0),
        )


@dataclass
class PatternMeta:
    """Pattern metadata - High-level patterns extracted from multiple cases"""
    pattern_id: str
    pattern_type: str                       # e.g., "age_related", "comorbidity_interaction"
    description: str                        # Pattern description
    frequency: int = 1                      # Occurrence frequency
    success_rate: float = 0.0                # Success rate when applying this pattern
    applicable_features: Dict[str, Any]     # Applicable feature conditions
    best_practices: List[str]                # Best practice recommendations
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "applicable_features": self.applicable_features,
            "best_practices": self.best_practices,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PatternMeta":
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            description=data["description"],
            frequency=data.get("frequency", 1),
            success_rate=data.get("success_rate", 0.0),
            applicable_features=data.get("applicable_features", {}),
            best_practices=data.get("best_practices", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
