"""Adapters for SoC, baseline, and TrialAgents-style melanoma decisions."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SIM_ROOT = ROOT / "m-agents" / "utils" / "rare_simulation"


ACTION_SPACE = [
    "Surgery (WLE)",
    "Surgery (WLE+SLNB)",
    "Surgery (WLE+SLNB) + Adjuvant: Targeted_Dual",
    "Surgery (WLE+SLNB) + Adjuvant: Immuno_Single",
    "Surgery + Adjuvant: Targeted_Dual",
    "Surgery + Adjuvant: Immuno_Single",
    "Neoadjuvant: Targeted_Dual + Surgery",
    "Neoadjuvant: Immuno_Single + Surgery",
    "Neoadjuvant: Immuno_Dual + Surgery",
    "Targeted_Dual",
    "Targeted Therapy (KIT Inhibitor)",
    "Immuno_Single",
    "Immuno_Dual",
    "Immuno_Single + Anti-Angiogenic + Immuno_Dual",
    "Chemo",
    "Supportive + Chemo",
    "Clinical Trial",
    "SRS + Targeted_Dual",
    "SRS + Targeted Therapy (KIT Inhibitor)",
    "SRS + Immuno_Dual",
    "SRS + Chemo",
]


def _load_function(module_path: Path, module_name: str, function_name: str) -> Callable:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def _legacy_soc_function() -> Callable:
    return _load_function(
        LEGACY_SIM_ROOT / "src" / "soc.py",
        "legacy_melanoma_soc",
        "recommend_melanoma_treatment",
    )


def _legacy_baseline_function() -> Callable:
    return _load_function(
        LEGACY_SIM_ROOT / "src" / "soc_baseline.py",
        "legacy_melanoma_soc_baseline",
        "recommend_melanoma_treatment_baseline",
    )


def _legacy_mas_agent_class() -> Callable:
    try:
        return _load_function(
            LEGACY_SIM_ROOT / "src" / "mas.py",
            "legacy_melanoma_mas",
            "MelanomaTreatmentAgent",
        )
    except ModuleNotFoundError as exc:
        if exc.name != "faker":
            raise
        # The legacy MAS module imports Faker at module scope, but the melanoma
        # treatment agent does not use it. Provide a tiny stub so the agent class
        # can be loaded without adding a simulation-only dependency.
        faker_stub = types.ModuleType("faker")
        faker_stub.Faker = lambda *args, **kwargs: None
        sys.modules.setdefault("faker", faker_stub)
        return _load_function(
            LEGACY_SIM_ROOT / "src" / "mas.py",
            "legacy_melanoma_mas",
            "MelanomaTreatmentAgent",
        )


def map_params_to_soc(params: Dict[str, Any]) -> Dict[str, Any]:
    """Map richer generated params to the legacy SoC function interface."""

    return {
        "stage": params.get("stage"),
        "braf_mutation": params.get("braf_mutation", False),
        "ecog": params.get("ecog", 0),
        "tumor_thickness": params.get("tumor_thickness"),
        "ulceration": params.get("ulceration"),
        "mitotic_rate": params.get("mitotic_rate"),
        "resectable": params.get("resectable"),
        "pd_l1_positive": params.get("pd_l1_positive"),
        "tumor_burden": params.get("tumor_burden"),
        "brain_metastases": params.get("brain_metastases", False),
        "previous_treatments": params.get("previous_treatments", []),
        "subtype": params.get("subtype", "cutaneous"),
    }


def _result_to_decision(result: Dict[str, Any], fallback: str = "Supportive + Chemo") -> Tuple[str, str]:
    if isinstance(result, dict) and result.get("recommended_treatment"):
        return str(result["recommended_treatment"]), str(result.get("guideline_basis", ""))
    if isinstance(result, dict) and result.get("error"):
        return fallback, str(result["error"])
    return fallback, "No valid recommendation returned."


class SocAdapter:
    """Wrapper around the legacy upgraded SoC rule tree."""

    def __init__(self) -> None:
        self._fn = _legacy_soc_function()

    def recommend(self, params: Dict[str, Any], text_profile: Optional[str] = None) -> Tuple[str, str]:
        del text_profile
        try:
            return _result_to_decision(self._fn(**map_params_to_soc(params)))
        except Exception as exc:
            return "Supportive + Chemo", f"SoC adapter error: {exc}"


class BaselineSocAdapter:
    """Wrapper around the legacy simplified baseline rule tree."""

    def __init__(self) -> None:
        self._fn = _legacy_baseline_function()

    def recommend(self, params: Dict[str, Any], text_profile: Optional[str] = None) -> Tuple[str, str]:
        del text_profile
        try:
            return _result_to_decision(self._fn(**map_params_to_soc(params)))
        except Exception as exc:
            return "Supportive + Chemo", f"Baseline adapter error: {exc}"


class TrialAgentsAdapter:
    """Wrapper around the legacy MAS melanoma treatment agent."""

    def __init__(self, mode: str = "rule_based", llm_model: str = "deepseek-chat",
                 doctor_num: int = 4, max_round: int = 1):
        """
        Initialize the TrialAgents adapter.

        :param mode: Recommendation mode, one of "rule_based", "single_agent", "multi_agent"
        :param llm_model: LLM model name
        :param doctor_num: Number of specialists in multi-agent mode
        :param max_round: Maximum number of discussion rounds in multi-agent mode
        """
        self.mode = mode
        self.llm_model = llm_model
        self.doctor_num = doctor_num
        self.max_round = max_round
        agent_cls = _legacy_mas_agent_class()
        self.agent = agent_cls(mode=mode, llm_model=llm_model,
                               doctor_num=doctor_num, max_round=max_round)

    def recommend(self, params: Dict[str, Any], text_profile: Optional[str] = None) -> Tuple[str, str]:
        try:
            raw_decision = self.agent.recommend_treatment(dict(params), text_profile)
            decision = extract_action(str(raw_decision))
            basis = f"MelanomaTreatmentAgent ({self.mode} mode) recommendation"
            if decision != raw_decision:
                basis += f"; raw_agent_output={raw_decision}"
            if text_profile:
                basis += "; narrative_profile_used=True"
            return decision, basis
        except Exception as exc:
            return "Supportive + Chemo", f"TrialAgents MAS adapter error: {exc}"

def extract_action(text: str) -> str:
    """Extract the longest known action contained in a long-form agent answer."""

    lower_text = text.lower()
    for action in sorted(ACTION_SPACE, key=len, reverse=True):
        if action.lower() in lower_text:
            return action
    return text.strip() or "Supportive + Chemo"


__all__ = [
    "ACTION_SPACE",
    "SocAdapter",
    "BaselineSocAdapter",
    "TrialAgentsAdapter",
    "extract_action",
    "map_params_to_soc",
]
