"""
TrialAgents Memory Bank - 数据结构定义

定义错误案例、成功案例和模式元数据的存储结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ErrorType(Enum):
    """错误类型枚举"""
    UNDERESTIMATION = "underestimation"       # 低估风险
    OVERESTIMATION = "overestimation"        # 高估风险
    MISDIAGNOSIS = "misdiagnosis"           # 误诊
    WRONG_BASIS = "wrong_basis"             # 依据错误
    INCOMPLETE_ANALYSIS = "incomplete_analysis"  # 分析不完整
    CONTRADICTION = "contradiction"          # 逻辑矛盾
    UNKNOWN = "unknown"                      # 未知类型


@dataclass
class PatientProfile:
    """患者特征摘要（用于相似度检索）"""
    patient_id: str
    dataset: str
    group: str                               # control/experimental
    key_features: Dict[str, Any]            # 关键特征摘要
    prediction: float                        # 预测值
    ground_truth: Optional[int] = None      # 真实标签

    def to_text(self) -> str:
        """转换为检索文本"""
        features_str = ", ".join([f"{k}={v}" for k, v in self.key_features.items()])
        return f"Patient {self.patient_id} ({self.dataset}/{self.group}): {features_str}, prediction={self.prediction}"


@dataclass
class ErrorCase:
    """错误案例"""
    case_id: str
    dataset: str
    patient_profile: PatientProfile
    clinician_analysis: str                    # 出错的分析内容
    error_type: ErrorType
    root_cause: str                         # 错误根因（LLM生成）
    correction: str                         # 正确分析
    improvement_hint: str                    # 改进提示
    timestamp: datetime = field(default_factory=datetime.now)
    frequency: int = 1                      # 出现频次
    clinician_index: int = 0                   # 出错的clinician索引
    prediction: float = 0.0                 # 错误预测值
    ground_truth: int = 0                   # 真实标签

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
        """从字典创建"""
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
    """成功案例"""
    case_id: str
    dataset: str
    patient_profile: PatientProfile
    reasoning_chain: str                   # 成功的推理链
    outcome: str                            # 预测结果描述
    confidence: float                       # 置信度
    key_triggers: List[str]                 # 有效触发点
    timestamp: datetime = field(default_factory=datetime.now)
    clinician_index: int = 0                   # clinician索引

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
        """从字典创建"""
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
    """模式元数据 - 从多个案例中提取的高层模式"""
    pattern_id: str
    pattern_type: str                       # e.g., "age_related", "comorbidity_interaction"
    description: str                        # 模式描述
    frequency: int = 1                      # 出现频次
    success_rate: float = 0.0                # 应用此模式时的成功率
    applicable_features: Dict[str, Any]     # 适用特征条件
    best_practices: List[str]                # 最佳实践建议
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
