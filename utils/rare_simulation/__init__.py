"""Rare melanoma virtual simulation package for TrialAgents."""

from .generator import generate_patients, summarize_cohort
from .oracle import MelanomaOutcomeOracle, parse_treatment

__all__ = [
    "MelanomaOutcomeOracle",
    "generate_patients",
    "parse_treatment",
    "summarize_cohort",
]
