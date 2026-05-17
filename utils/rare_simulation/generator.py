"""Patient generator for borderline cutaneous melanoma simulation."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np


def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Latin hypercube samples in [0, 1]."""

    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(0, 1, size=(n, d))
    a, b = cut[:n], cut[1:]
    pts = u * (b - a)[:, None] + a[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def _choice_by_threshold(value: float, labels: List[str], weights: List[float]) -> str:
    cumulative = np.cumsum(weights)
    for label, threshold in zip(labels, cumulative):
        if value <= threshold:
            return label
    return labels[-1]


def _sample_ethnicity(value: float) -> str:
    return _choice_by_threshold(value, ["Caucasian", "Asian", "Hispanic", "African"], [0.56, 0.25, 0.12, 0.07])


def _sample_stage(value: float) -> str:
    return _choice_by_threshold(value, ["0", "I", "II", "III", "IV"], [0.04, 0.30, 0.24, 0.22, 0.20])


def _sample_subtype(ethnicity: str, value: float, enrich_edge_cases: bool) -> str:
    del ethnicity, value, enrich_edge_cases
    return "cutaneous"

    # Keep the previous subtype sampling logic here for future non-cutaneous
    # cohorts. Current experiments intentionally specialize in cutaneous melanoma.
    # if ethnicity == "Asian":
    #     labels = ["acral", "cutaneous", "mucosal"]
    #     weights = [0.44, 0.40, 0.16] if enrich_edge_cases else [0.30, 0.55, 0.15]
    # elif ethnicity == "African":
    #     labels = ["acral", "cutaneous", "mucosal"]
    #     weights = [0.36, 0.52, 0.12] if enrich_edge_cases else [0.25, 0.65, 0.10]
    # else:
    #     labels = ["cutaneous", "acral", "mucosal"]
    #     weights = [0.78, 0.12, 0.10] if enrich_edge_cases else [0.86, 0.08, 0.06]
    # return _choice_by_threshold(value, labels, weights)


def _sample_ecog(stage: str, value: float) -> int:
    if stage in {"0", "I", "II"}:
        return int(np.digitize(value, [0.65, 0.90, 0.98]))
    return int(np.digitize(value, [0.38, 0.72, 0.92]))


def _sample_thickness(stage: str, value: float) -> float:
    if stage == "0":
        return 0.0
    if stage == "I":
        # Deliberately includes T1b borderline area near 0.8-1.0 mm.
        return float(round(0.2 + value * 1.0, 2))
    if stage == "II":
        return float(round(1.01 + value * 4.2, 2))
    if stage == "III":
        return float(round(1.5 + value * 5.5, 2))
    return float(round(2.0 + value * 6.0, 2))


def _sample_mitotic_rate(stage: str, thickness: float, value: float) -> int:
    if stage == "0":
        return 0
    if thickness <= 1.0:
        return int(np.digitize(value, [0.45, 0.70, 0.88, 0.96]))
    if stage in {"I", "II"}:
        return int(np.digitize(value, [0.25, 0.50, 0.75, 0.90, 0.97]))
    return int(np.digitize(value, [0.15, 0.35, 0.58, 0.80, 0.94])) + 1


def _pd_l1_cps(value: float) -> float:
    return float(round(value * 20.0, 2))


def _tmb(value: float, subtype: str) -> float:
    del subtype
    max_tmb = 38.0
    # Previous non-cutaneous setting:
    # max_tmb = 38.0 if subtype == "cutaneous" else 18.0
    return float(round(1.0 + value * (max_tmb - 1.0), 1))


def _build_edge_case_tags(params: Dict) -> List[str]:
    tags: List[str] = []
    stage = params["stage"]
    thickness = float(params.get("tumor_thickness", 0.0))
    mitotic = int(params.get("mitotic_rate", 0))

    # Non-cutaneous cohorts can restore this edge tag together with subtype
    # sampling and KIT enrichment.
    # if params.get("kit_mutation") and params.get("subtype") in {"acral", "mucosal"}:
    #     tags.append("kit_acral_mucosal")
    if params.get("autoimmune_disease"):
        tags.append("autoimmune_ici_contraindication")
    if params.get("steroid_use"):
        tags.append("baseline_steroid_use")
    if params.get("brain_metastases") and params.get("symptomatic_brain_metastases"):
        tags.append("symptomatic_brain_needs_srs")
    if params.get("liver_metastases"):
        tags.append("liver_mets_avoid_single_io")
    if params.get("relapse_during_adjuvant"):
        tags.append("relapse_on_adjuvant_pd1")
    if stage == "I" and 0.75 <= thickness <= 1.05 and (params.get("ulceration") or mitotic >= 1):
        tags.append("t1b_slnb_borderline")
    if stage == "II" and (params.get("ulceration") or thickness >= 4.0 or mitotic >= 5):
        tags.append("stage_ii_high_risk_adjuvant")
    if params.get("borderline_resectable"):
        tags.append("borderline_resectable_iii")
    if params.get("tumor_burden") == "high":
        tags.append("high_tumor_burden")
    if params.get("organ_dysfunction"):
        tags.append("organ_dysfunction_limits_treatment")
    return tags


def _previous_treatments(prior_ici: bool, prior_braf_mek: bool) -> List[str]:
    treatments: List[str] = []
    if prior_ici:
        treatments.append("Immuno_Single")
    if prior_braf_mek:
        treatments.append("Targeted_Dual")
    return treatments


def _map_params(sample: np.ndarray, rng: np.random.Generator, idx: int, enrich_edge_cases: bool) -> Dict:
    age = int(round(22 + sample[0] * 66))
    sex = "male" if sample[1] < 0.55 else "female"
    ethnicity = _sample_ethnicity(sample[2])
    stage = _sample_stage(sample[3])
    subtype = _sample_subtype(ethnicity, sample[4], enrich_edge_cases)
    thickness = _sample_thickness(stage, sample[5])
    ulceration = bool(sample[6] > (0.62 if stage in {"I", "II"} else 0.50))
    mitotic_rate = _sample_mitotic_rate(stage, thickness, sample[7])
    pd_l1 = _pd_l1_cps(sample[8])
    tmb = _tmb(sample[9], subtype)
    ldh_ratio = float(round(0.65 + sample[10] * 1.2, 2))
    ecog = _sample_ecog(stage, sample[11])

    brain_metastases = False
    symptomatic_brain_metastases = False
    liver_metastases = False
    bone_metastases = False
    visceral_metastases_count = 0
    in_transit_metastases = False

    if stage == "III":
        in_transit_metastases = bool(sample[12] < 0.22)
    elif stage == "IV":
        brain_metastases = bool(sample[12] < (0.34 if enrich_edge_cases else 0.28))
        symptomatic_brain_metastases = bool(brain_metastases and sample[13] < 0.55)
        liver_metastases = bool(sample[14] < 0.30)
        bone_metastases = bool(sample[15] < 0.22)
        visceral_metastases_count = int(liver_metastases) + int(bone_metastases) + int(sample[16] < 0.24)
        ldh_boost = 1.0 if liver_metastases or visceral_metastases_count >= 2 else 0.4
        ldh_ratio = float(round(0.75 + sample[10] * (2.7 if ldh_boost else 1.1), 2))

    kit_mutation = False
    braf_mutation = False
    nras_mutation = False
    kit_exon: Optional[str] = None

    # Cutaneous-focused cohort: do not enrich acral/mucosal KIT-mutant cases.
    braf_cutoff = 0.43
    nras_cutoff = braf_cutoff + 0.21
    genotype_value = sample[17]
    if genotype_value < braf_cutoff:
        braf_mutation = True
    elif genotype_value < nras_cutoff:
        nras_mutation = True

    # Previous non-cutaneous/KIT enrichment logic:
    # if subtype in {"acral", "mucosal"} and sample[17] < (0.20 if enrich_edge_cases else 0.14):
    #     kit_mutation = True
    #     kit_exon = _choice_by_threshold(sample[18], ["exon 11 L576P", "exon 13 K642E", "exon 17"], [0.48, 0.34, 0.18])
    # else:
    #     braf_cutoff = 0.16 if subtype in {"acral", "mucosal"} else 0.43
    #     nras_cutoff = braf_cutoff + (0.16 if subtype in {"acral", "mucosal"} else 0.21)
    #     genotype_value = sample[17]
    #     if genotype_value < braf_cutoff:
    #         braf_mutation = True
    #     elif genotype_value < nras_cutoff:
    #         nras_mutation = True

    autoimmune_disease = bool(sample[19] < (0.14 if enrich_edge_cases else 0.09))
    steroid_use = bool(autoimmune_disease and sample[20] < 0.45)
    organ_dysfunction = bool(sample[21] < 0.08 or (age >= 78 and sample[21] < 0.18))

    prior_adjuvant_therapy = bool(stage in {"II", "III", "IV"} and sample[22] < 0.30)
    relapse_during_adjuvant = bool(prior_adjuvant_therapy and sample[22] < 0.12)
    prior_ici_exposure = bool(prior_adjuvant_therapy and sample[23] < 0.75)
    prior_braf_mek_exposure = bool(prior_adjuvant_therapy and braf_mutation and sample[23] >= 0.75)
    previous_treatments = _previous_treatments(prior_ici_exposure, prior_braf_mek_exposure)

    resectable = None
    borderline_resectable = False
    if stage == "III":
        resectable = bool(sample[24] < 0.62)
        borderline_resectable = bool(resectable and sample[24] > 0.48)

    tumor_burden_score = 0.0
    if stage == "IV":
        tumor_burden_score = min(
            1.0,
            0.25
            + 0.15 * visceral_metastases_count
            + (0.20 if brain_metastases else 0.0)
            + (0.20 if ldh_ratio > 1.5 else 0.0)
            + (0.10 if ecog >= 2 else 0.0),
        )
    elif stage == "III":
        tumor_burden_score = min(1.0, 0.25 + (0.25 if in_transit_metastases else 0.0) + (0.20 if not resectable else 0.0))

    tumor_burden = "high" if tumor_burden_score >= 0.60 else "low"

    params = {
        "id": f"RD-{idx:04d}-{uuid.uuid4().hex[:8]}",
        "age": age,
        "sex": sex,
        "ethnicity": ethnicity,
        "stage": stage,
        "subtype": subtype,
        "tumor_thickness": thickness,
        "ulceration": ulceration,
        "mitotic_rate": mitotic_rate,
        "genotype": "KIT" if kit_mutation else ("BRAF" if braf_mutation else ("NRAS" if nras_mutation else "WT")),
        "braf_mutation": braf_mutation,
        "nras_mutation": nras_mutation,
        "kit_mutation": kit_mutation,
        "kit_exon": kit_exon,
        "pd_l1_cps": pd_l1,
        "pd_l1_positive": bool(pd_l1 >= 1.0),
        "tmb": tmb,
        "ldh_ratio": ldh_ratio,
        "ecog": ecog,
        "brain_metastases": brain_metastases,
        "symptomatic_brain_metastases": symptomatic_brain_metastases,
        "liver_metastases": liver_metastases,
        "bone_metastases": bone_metastases,
        "visceral_metastases_count": visceral_metastases_count,
        "in_transit_metastases": in_transit_metastases,
        "autoimmune_disease": autoimmune_disease,
        "steroid_use": steroid_use,
        "organ_dysfunction": organ_dysfunction,
        "prior_adjuvant_therapy": prior_adjuvant_therapy,
        "relapse_during_adjuvant": relapse_during_adjuvant,
        "prior_ici_exposure": prior_ici_exposure,
        "prior_braf_mek_exposure": prior_braf_mek_exposure,
        "previous_treatments": previous_treatments,
        "resectable": resectable,
        "borderline_resectable": borderline_resectable,
        "tumor_burden_score": float(round(tumor_burden_score, 2)),
        "tumor_burden": tumor_burden,
    }
    params["edge_case_tags"] = _build_edge_case_tags(params)
    return params


def build_text_profile(params: Dict) -> str:
    stage = params["stage"]
    stage_text = {
        "0": "melanoma in situ (stage 0)",
        "I": f"localized stage I melanoma, Breslow {params['tumor_thickness']} mm",
        "II": f"high/localized stage II melanoma, Breslow {params['tumor_thickness']} mm",
        "III": f"regional stage III melanoma, Breslow {params['tumor_thickness']} mm",
        "IV": f"metastatic stage IV melanoma, Breslow {params['tumor_thickness']} mm",
    }[stage]

    if params.get("in_transit_metastases"):
        stage_text += ", with in-transit metastases"
    if stage == "III":
        if params.get("borderline_resectable"):
            stage_text += ", borderline resectable"
        elif params.get("resectable") is True:
            stage_text += ", resectable"
        elif params.get("resectable") is False:
            stage_text += ", unresectable"

    subtype_text = {
        "cutaneous": "cutaneous melanoma",
        # "acral": "acral melanoma involving palms/soles/nail unit",
        # "mucosal": "mucosal melanoma involving head/neck or anorectal mucosa",
    }[params["subtype"]]

    genotype = params["genotype"]
    if genotype == "KIT":
        genomics = f"KIT mutant ({params.get('kit_exon')})"
    elif genotype == "BRAF":
        genomics = "BRAF V600E/K mutant"
    elif genotype == "NRAS":
        genomics = "NRAS mutant, BRAF/KIT wild type"
    else:
        genomics = "BRAF/NRAS/KIT wild type"

    ulcer = "ulcerated" if params["ulceration"] else "non-ulcerated"
    brain = "no brain metastases"
    if params.get("brain_metastases"):
        brain = "symptomatic brain metastases requiring local therapy/SRS" if params.get("symptomatic_brain_metastases") else "asymptomatic brain metastases"

    metastases = []
    if params.get("liver_metastases"):
        metastases.append("liver metastases")
    if params.get("bone_metastases"):
        metastases.append("bone metastases")
    metastasis_text = ", ".join(metastases) if metastases else "no liver or bone metastases documented"

    history = []
    if params.get("autoimmune_disease"):
        history.append("active autoimmune disease making checkpoint immunotherapy high-risk")
    if params.get("steroid_use"):
        history.append("baseline systemic steroid use")
    if params.get("organ_dysfunction"):
        history.append("clinically significant organ dysfunction limiting cytotoxic therapy")
    if params.get("prior_adjuvant_therapy"):
        if params.get("relapse_during_adjuvant"):
            history.append("progression during or within 6 months of adjuvant anti-PD-1 therapy")
        else:
            history.append("remote prior adjuvant therapy with recurrence after 6 months")
    history_text = "; ".join(history) if history else "no major autoimmune disease or prior systemic therapy"

    return (
        f"{params['age']}yo {params['sex']} ({params['ethnicity']}), {stage_text}, {subtype_text}, {ulcer}. "
        f"Mitotic rate {params['mitotic_rate']}/mm2. Genomics: {genomics}. "
        f"Biomarkers: PD-L1 CPS {params['pd_l1_cps']}, TMB {params['tmb']} muts/Mb, LDH {params['ldh_ratio']}x ULN. "
        f"Clinical status: ECOG {params['ecog']}, tumor burden {params['tumor_burden']} "
        f"(score {params['tumor_burden_score']}), {brain}, {metastasis_text}. "
        f"History: {history_text}. Edge case tags: {', '.join(params['edge_case_tags']) or 'none'}."
    )


def generate_patients(n: int = 100, seed: int = 42, enrich_edge_cases: bool = True) -> List[Dict]:
    """Generate a deterministic cohort for virtual melanoma trials."""

    rng = np.random.default_rng(seed)
    samples = _latin_hypercube(n, 25, rng)
    patients: List[Dict] = []
    for idx, sample in enumerate(samples):
        params = _map_params(sample, rng, idx, enrich_edge_cases)
        patients.append(
            {
                "hidden_params": params,
                "text_profile": build_text_profile(params),
                "edge_case_tags": params["edge_case_tags"],
            }
        )
    return patients


def summarize_cohort(patients: List[Dict]) -> Dict:
    """Return lightweight distribution checks for generated cohorts."""

    summary: Dict[str, Dict[str, int]] = {
        "stage": {},
        "subtype": {},
        "genotype": {},
        "edge_case_tags": {},
    }
    for patient in patients:
        params = patient["hidden_params"]
        for key in ("stage", "subtype", "genotype"):
            value = params.get(key, "unknown")
            summary[key][value] = summary[key].get(value, 0) + 1
        for tag in params.get("edge_case_tags", []):
            summary["edge_case_tags"][tag] = summary["edge_case_tags"].get(tag, 0) + 1
    return summary


__all__ = ["generate_patients", "build_text_profile", "summarize_cohort"]
