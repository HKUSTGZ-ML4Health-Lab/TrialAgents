"""Evidence-anchored oracle for melanoma virtual trials.

The oracle estimates a deterministic 12-month event-free probability. For
metastatic or unresectable disease the endpoint is interpreted as PFS; for
resected/localized settings it is interpreted as RFS/EFS. The model is not a
clinical predictor. It is an auditable in-silico judge that uses multiplicative
hazard ratios rather than additive probability bonuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EvidenceEntry:
    """A compact provenance record for one hazard ratio assumption."""

    key: str
    hr: float
    setting: str
    endpoint: str
    comparator: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class TreatmentProfile:
    """Structured treatment features parsed from a free-form decision string."""

    raw: str
    has_surgery: bool = False
    has_slnb: bool = False
    has_srs: bool = False
    has_wbrt: bool = False
    has_targeted_dual: bool = False
    has_kit_inhibitor: bool = False
    has_immuno_single: bool = False
    has_immuno_dual: bool = False
    has_anti_angiogenic: bool = False
    has_chemo: bool = False
    supportive_only: bool = False
    has_clinical_trial: bool = False


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def parse_treatment(treatment_decision: Optional[str]) -> TreatmentProfile:
    """Normalize SoC, baseline, MAS, and TrialAgents treatment strings.

    Existing modules emit mixed strings such as "Surgery + SRS +
    Targeted_Dual + Supportive" or "Targeted Therapy (KIT Inhibitor)". This
    parser converts them to comparable tags before the oracle applies HRs.
    """

    raw = treatment_decision or ""
    text = raw.lower()

    has_surgery = _contains_any(text, ["surgery", "wle", "wide excision", "excision"])
    has_slnb = _contains_any(text, ["slnb", "sentinel"])
    has_srs = _contains_any(text, ["srs", "stereotactic"])
    has_wbrt = _contains_any(text, ["wbrt", "whole brain"])
    has_kit_inhibitor = _contains_any(text, ["kit inhibitor", "imatinib", "nilotinib", "dasatinib"])
    has_targeted_dual = _contains_any(text, ["targeted_dual", "braf/mek", "braf-mek", "dabrafenib", "trametinib"])
    if not has_kit_inhibitor and _contains_any(text, ["targeted therapy", "targeted"]):
        has_targeted_dual = has_targeted_dual or "kit" not in text

    has_immuno_dual = _contains_any(text, ["immuno_dual", "dual io", "dual immuno", "nivo+ipi", "ipilimumab+nivolumab"])
    has_immuno_single = _contains_any(text, ["immuno_single", "anti-pd-1", "pd-1", "pembrolizumab", "nivolumab"])
    if "immunotherapy" in text and not has_immuno_dual:
        has_immuno_single = True

    has_chemo = _contains_any(text, ["chemo", "chemotherapy", "dacarbazine", "temozolomide"])
    supportive = _contains_any(text, ["supportive", "palliative", "best supportive"])
    has_clinical_trial = _contains_any(text, ["clinical trial", "trial"])

    active_tags = [
        has_surgery,
        has_srs,
        has_wbrt,
        has_targeted_dual,
        has_kit_inhibitor,
        has_immuno_single,
        has_immuno_dual,
        has_chemo,
        has_clinical_trial,
    ]

    return TreatmentProfile(
        raw=raw,
        has_surgery=has_surgery,
        has_slnb=has_slnb,
        has_srs=has_srs,
        has_wbrt=has_wbrt,
        has_targeted_dual=has_targeted_dual,
        has_kit_inhibitor=has_kit_inhibitor,
        has_immuno_single=has_immuno_single,
        has_immuno_dual=has_immuno_dual,
        has_anti_angiogenic=_contains_any(text, ["anti-angiogenic", "anti angiogenic", "bevacizumab", "lenvatinib"]),
        has_chemo=has_chemo,
        supportive_only=supportive and not any(active_tags),
        has_clinical_trial=has_clinical_trial,
    )


class MelanomaOutcomeOracle:
    """Deterministic 12-month PFS/RFS oracle using multiplicative HRs."""

    def __init__(self, eval_time_months: int = 12):
        if eval_time_months != 12:
            raise ValueError("This oracle is calibrated only for 12-month PFS/RFS.")
        self.eval_time = eval_time_months

        self.base_event_free_12 = {
            "rfs": {
                "0": {"cutaneous": 0.995, "acral": 0.990, "mucosal": 0.970},
                "I": {"cutaneous": 0.965, "acral": 0.940, "mucosal": 0.900},
                "II": {"cutaneous": 0.820, "acral": 0.760, "mucosal": 0.700},
                "III": {"cutaneous": 0.680, "acral": 0.580, "mucosal": 0.520},
            },
            "pfs": {
                "III": {"cutaneous": 0.470, "acral": 0.390, "mucosal": 0.350},
                "IV": {"cutaneous": 0.320, "acral": 0.260, "mucosal": 0.220},
            },
        }

        self.evidence: Dict[str, EvidenceEntry] = {
            "adjuvant_braf_mek": EvidenceEntry(
                key="adjuvant_braf_mek",
                hr=0.51,
                setting="Resected stage III BRAF V600 melanoma",
                endpoint="RFS",
                comparator="placebo",
                source="COMBI-AD 5-year analysis",
                note="Used for BRAF/MEK benefit in resected high-risk disease.",
            ),
            "advanced_braf_mek": EvidenceEntry(
                key="advanced_braf_mek",
                hr=0.45,
                setting="Unresectable/metastatic BRAF V600 melanoma",
                endpoint="PFS",
                comparator="historical chemotherapy/control",
                source="BRAF/MEK pivotal trial approximation",
                note="Conservative approximation for active BRAF/MEK therapy.",
            ),
            "pd1_advanced": EvidenceEntry(
                key="pd1_advanced",
                hr=0.57,
                setting="Advanced melanoma",
                endpoint="PFS",
                comparator="ipilimumab",
                source="KEYNOTE-006",
                note="Pembrolizumab PFS HR approximation.",
            ),
            "pd1_adjuvant_stage_ii": EvidenceEntry(
                key="pd1_adjuvant_stage_ii",
                hr=0.61,
                setting="Resected high-risk stage IIB/IIC melanoma",
                endpoint="RFS",
                comparator="placebo",
                source="KEYNOTE-716",
                note="Adjuvant anti-PD-1 RFS approximation.",
            ),
            "dual_io_advanced": EvidenceEntry(
                key="dual_io_advanced",
                hr=0.42,
                setting="Untreated unresectable stage III/IV melanoma",
                endpoint="PFS",
                comparator="ipilimumab",
                source="CheckMate 067",
                note="Nivolumab plus ipilimumab PFS HR approximation.",
            ),
            "kit_inhibitor": EvidenceEntry(
                key="kit_inhibitor",
                hr=0.48,
                setting="KIT-altered acral/mucosal melanoma",
                endpoint="PFS",
                comparator="non-targeted systemic therapy",
                source="Phase II imatinib studies",
                note="Benefit applies only to mutation-activated KIT edge cases.",
            ),
            "chemo": EvidenceEntry(
                key="chemo",
                hr=0.92,
                setting="Advanced melanoma",
                endpoint="PFS",
                comparator="supportive/historical control",
                source="Historical chemotherapy approximation",
                note="Small palliative disease-control effect.",
            ),
        }

        self.prognostic_hr = {
            "ecog_1": 1.20,
            "ecog_2": 1.65,
            "ecog_3": 2.40,
            "ldh_1_2x": 1.35,
            "ldh_gt2x": 1.85,
            "brain_mets": 1.45,
            "symptomatic_brain_extra": 1.25,
            "liver_mets": 1.40,
            "bone_mets": 1.20,
            "visceral_multi": 1.25,
            "high_burden": 1.30,
            "ulceration": 1.15,
            "thickness_gt4": 1.20,
            "mitotic_high": 1.10,
            "age_ge75": 1.15,
            "autoimmune": 1.10,
            "steroid_use": 1.20,
            "organ_dysfunction": 1.25,
            "relapse_on_adjuvant": 1.55,
            "acral": 1.12,
            "mucosal": 1.25,
            "nras": 1.12,
        }

        self.interaction_hr = {
            "ineffective_braf_targeting": 1.35,
            "ineffective_kit_targeting": 1.30,
            "brain_no_local": 1.30,
            "brain_targeted_without_local": 1.15,
            "brain_with_local": 0.88,
            "liver_single_io": 1.25,
            "dual_io_high_burden": 0.92,
            "autoimmune_immuno": 3.20,
            "steroid_immuno": 1.25,
            "ecog2_dual_io": 1.30,
            "age_chemo": 1.20,
            "anti_angiogenic_acral": 0.92,
            "clinical_trial_edge": 0.96,
        }

    def _endpoint(self, patient: Dict) -> str:
        stage = patient.get("stage")
        if stage in {"0", "I", "II"}:
            return "rfs"
        if stage == "III" and patient.get("resectable", False):
            return "rfs"
        return "pfs"

    def _base_probability(self, patient: Dict) -> Tuple[float, str]:
        endpoint = self._endpoint(patient)
        stage = str(patient.get("stage"))
        subtype = patient.get("subtype", "cutaneous")
        table = self.base_event_free_12[endpoint]
        if stage not in table:
            stage = "IV" if endpoint == "pfs" else "III"
        if subtype not in table[stage]:
            subtype = "cutaneous"
        return table[stage][subtype], endpoint

    @staticmethod
    def _multiply(risk: float, hr: float, label: str, log: List[str]) -> float:
        new_risk = risk * hr
        log.append(f"{label}: HR {hr:.3f} -> cumulative HR {new_risk:.3f}")
        return new_risk

    def _apply_prognostic_hrs(self, patient: Dict, risk: float, log: List[str]) -> float:
        ecog = int(patient.get("ecog", 0) or 0)
        if ecog == 1:
            risk = self._multiply(risk, self.prognostic_hr["ecog_1"], "ECOG 1", log)
        elif ecog == 2:
            risk = self._multiply(risk, self.prognostic_hr["ecog_2"], "ECOG 2", log)
        elif ecog >= 3:
            risk = self._multiply(risk, self.prognostic_hr["ecog_3"], "ECOG >=3", log)

        ldh = float(patient.get("ldh_ratio", 1.0) or 1.0)
        if ldh > 2.0:
            risk = self._multiply(risk, self.prognostic_hr["ldh_gt2x"], "LDH >2x ULN", log)
        elif ldh > 1.0:
            risk = self._multiply(risk, self.prognostic_hr["ldh_1_2x"], "LDH 1-2x ULN", log)

        if patient.get("brain_metastases"):
            risk = self._multiply(risk, self.prognostic_hr["brain_mets"], "Brain metastases", log)
            if patient.get("symptomatic_brain_metastases"):
                risk = self._multiply(risk, self.prognostic_hr["symptomatic_brain_extra"], "Symptomatic CNS disease", log)
        if patient.get("liver_metastases"):
            risk = self._multiply(risk, self.prognostic_hr["liver_mets"], "Liver metastases", log)
        if patient.get("bone_metastases"):
            risk = self._multiply(risk, self.prognostic_hr["bone_mets"], "Bone metastases", log)
        if int(patient.get("visceral_metastases_count", 0) or 0) >= 2:
            risk = self._multiply(risk, self.prognostic_hr["visceral_multi"], "Multiple visceral sites", log)
        if float(patient.get("tumor_burden_score", 0.0) or 0.0) >= 0.70 or patient.get("tumor_burden") == "high":
            risk = self._multiply(risk, self.prognostic_hr["high_burden"], "High tumor burden", log)

        if patient.get("ulceration") and patient.get("stage") in {"I", "II", "III"}:
            risk = self._multiply(risk, self.prognostic_hr["ulceration"], "Ulceration", log)
        if float(patient.get("tumor_thickness", 0.0) or 0.0) > 4.0:
            risk = self._multiply(risk, self.prognostic_hr["thickness_gt4"], "Breslow >4mm", log)
        if int(patient.get("mitotic_rate", 0) or 0) >= 5:
            risk = self._multiply(risk, self.prognostic_hr["mitotic_high"], "High mitotic rate", log)

        if int(patient.get("age", 0) or 0) >= 75:
            risk = self._multiply(risk, self.prognostic_hr["age_ge75"], "Age >=75", log)
        if patient.get("autoimmune_disease"):
            risk = self._multiply(risk, self.prognostic_hr["autoimmune"], "Active autoimmune disease", log)
        if patient.get("steroid_use"):
            risk = self._multiply(risk, self.prognostic_hr["steroid_use"], "Baseline steroid use", log)
        if patient.get("organ_dysfunction"):
            risk = self._multiply(risk, self.prognostic_hr["organ_dysfunction"], "Organ dysfunction", log)
        if patient.get("relapse_during_adjuvant"):
            risk = self._multiply(risk, self.prognostic_hr["relapse_on_adjuvant"], "Relapse on adjuvant therapy", log)

        subtype = patient.get("subtype")
        if subtype == "acral":
            risk = self._multiply(risk, self.prognostic_hr["acral"], "Acral subtype", log)
        elif subtype == "mucosal":
            risk = self._multiply(risk, self.prognostic_hr["mucosal"], "Mucosal subtype", log)
        if patient.get("nras_mutation"):
            risk = self._multiply(risk, self.prognostic_hr["nras"], "NRAS mutation", log)

        return risk

    def _apply_treatment_hrs(
        self,
        patient: Dict,
        treatment: TreatmentProfile,
        endpoint: str,
        risk: float,
        log: List[str],
        evidence_used: List[EvidenceEntry],
    ) -> float:
        stage = patient.get("stage")
        braf = bool(patient.get("braf_mutation"))
        kit = bool(patient.get("kit_mutation"))

        if treatment.has_targeted_dual:
            if braf:
                key = "adjuvant_braf_mek" if endpoint == "rfs" else "advanced_braf_mek"
                evidence = self.evidence[key]
                risk = self._multiply(risk, evidence.hr, "BRAF V600 + BRAF/MEK therapy", log)
                evidence_used.append(evidence)
            else:
                risk = self._multiply(risk, self.interaction_hr["ineffective_braf_targeting"], "BRAF/MEK without BRAF V600", log)

        if treatment.has_kit_inhibitor:
            if kit:
                evidence = self.evidence["kit_inhibitor"]
                risk = self._multiply(risk, evidence.hr, "KIT alteration + KIT inhibitor", log)
                evidence_used.append(evidence)
            else:
                risk = self._multiply(risk, self.interaction_hr["ineffective_kit_targeting"], "KIT inhibitor without KIT alteration", log)

        if treatment.has_immuno_dual:
            evidence = self.evidence["dual_io_advanced"]
            risk = self._multiply(risk, evidence.hr, "Dual checkpoint immunotherapy", log)
            evidence_used.append(evidence)
        elif treatment.has_immuno_single:
            key = "pd1_adjuvant_stage_ii" if endpoint == "rfs" and stage in {"II", "III"} else "pd1_advanced"
            evidence = self.evidence[key]
            risk = self._multiply(risk, evidence.hr, "Single-agent anti-PD-1", log)
            evidence_used.append(evidence)

        if treatment.has_chemo:
            evidence = self.evidence["chemo"]
            risk = self._multiply(risk, evidence.hr, "Chemotherapy", log)
            evidence_used.append(evidence)

        if treatment.has_clinical_trial and patient.get("edge_case_tags"):
            risk = self._multiply(risk, self.interaction_hr["clinical_trial_edge"], "Clinical trial option for edge case", log)

        return risk

    def _apply_interactions(self, patient: Dict, treatment: TreatmentProfile, risk: float, log: List[str]) -> float:
        has_cns_local = treatment.has_srs or treatment.has_wbrt or (treatment.has_surgery and patient.get("brain_metastases"))
        if patient.get("brain_metastases"):
            if has_cns_local:
                risk = self._multiply(risk, self.interaction_hr["brain_with_local"], "CNS local therapy present", log)
            elif patient.get("symptomatic_brain_metastases"):
                risk = self._multiply(risk, self.interaction_hr["brain_no_local"], "Symptomatic CNS disease without local therapy", log)
            if treatment.has_targeted_dual and not has_cns_local:
                risk = self._multiply(risk, self.interaction_hr["brain_targeted_without_local"], "Targeted therapy for CNS disease without local control", log)

        if patient.get("liver_metastases") and treatment.has_immuno_single and not treatment.has_immuno_dual:
            risk = self._multiply(risk, self.interaction_hr["liver_single_io"], "Liver metastases + single-agent IO", log)
        if treatment.has_immuno_dual and (patient.get("tumor_burden") == "high" or float(patient.get("tumor_burden_score", 0.0) or 0.0) >= 0.70):
            risk = self._multiply(risk, self.interaction_hr["dual_io_high_burden"], "Dual IO in high-burden disease", log)
        if patient.get("autoimmune_disease") and (treatment.has_immuno_single or treatment.has_immuno_dual):
            risk = self._multiply(risk, self.interaction_hr["autoimmune_immuno"], "Autoimmune disease + immunotherapy toxicity risk", log)
        if patient.get("steroid_use") and (treatment.has_immuno_single or treatment.has_immuno_dual):
            risk = self._multiply(risk, self.interaction_hr["steroid_immuno"], "Steroid use + immunotherapy", log)
        if int(patient.get("ecog", 0) or 0) >= 2 and treatment.has_immuno_dual:
            risk = self._multiply(risk, self.interaction_hr["ecog2_dual_io"], "ECOG >=2 + dual IO", log)
        if int(patient.get("age", 0) or 0) >= 75 and treatment.has_chemo:
            risk = self._multiply(risk, self.interaction_hr["age_chemo"], "Age >=75 + chemotherapy", log)
        if patient.get("subtype") == "acral" and treatment.has_anti_angiogenic:
            risk = self._multiply(risk, self.interaction_hr["anti_angiogenic_acral"], "Acral subtype + anti-angiogenic combination", log)

        return risk

    @staticmethod
    def _bound_probability(probability: float) -> float:
        return max(0.01, min(0.995, probability))

    def calculate_event_free_probability(self, patient_params: Dict, treatment_decision: str) -> Dict:
        base_probability, endpoint = self._base_probability(patient_params)
        treatment = parse_treatment(treatment_decision)
        log: List[str] = []
        evidence_used: List[EvidenceEntry] = []

        risk = 1.0
        log.append(f"Base 12-month {endpoint.upper()} probability: {base_probability:.3f}")
        risk = self._apply_prognostic_hrs(patient_params, risk, log)
        risk = self._apply_treatment_hrs(patient_params, treatment, endpoint, risk, log, evidence_used)
        risk = self._apply_interactions(patient_params, treatment, risk, log)

        final_probability = self._bound_probability(base_probability ** risk)
        evidence_lines = [
            f"- {entry.key}: HR={entry.hr:.2f}, {entry.endpoint}, {entry.source}, comparator={entry.comparator}"
            for entry in evidence_used
        ]
        if not evidence_lines:
            evidence_lines = ["- No active evidence HR applied; decision treated as local/supportive or unmatched."]

        full_log = (
            f"=== Melanoma Oracle: 12-month {endpoint.upper()} ===\n"
            f"Patient: {patient_params.get('id', 'unknown')}\n"
            f"Treatment: {treatment.raw}\n"
            f"Parsed treatment: {treatment}\n"
            + "\n".join(log)
            + f"\nFinal cumulative HR: {risk:.3f}\n"
            + f"Final 12-month {endpoint.upper()} probability: {final_probability:.3f}\n"
            + "Evidence used:\n"
            + "\n".join(evidence_lines)
        )

        return {
            "patient_id": patient_params.get("id", "unknown"),
            "endpoint": endpoint,
            "eval_time_months": self.eval_time,
            "treatment_decision": treatment_decision,
            "parsed_treatment": treatment.__dict__,
            "base_probability": base_probability,
            "cumulative_hr": risk,
            "event_free_probability": final_probability,
            "pfs_12": final_probability,
            "survival_probability": final_probability,
            "evidence": [entry.__dict__ for entry in evidence_used],
            "evaluation_log": full_log,
        }

    def calculate_survival_probability(self, patient_params: Dict, treatment_decision: str) -> Dict:
        """Backward-compatible alias used by the old simulation loop."""

        return self.calculate_event_free_probability(patient_params, treatment_decision)


__all__ = [
    "EvidenceEntry",
    "TreatmentProfile",
    "parse_treatment",
    "MelanomaOutcomeOracle",
]
