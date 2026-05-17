"""Run a rare/borderline melanoma virtual trial.

This entry point keeps the simulation independent from the legacy
``m-agents/utils/rare_simulation`` directory while reusing its SoC rule trees
through adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .adapters import BaselineSocAdapter, SocAdapter, TrialAgentsAdapter
    from .generator import generate_patients, summarize_cohort
    from .oracle import MelanomaOutcomeOracle, parse_treatment
except ImportError:  # pragma: no cover - allows `python main.py` from this dir.
    from adapters import BaselineSocAdapter, SocAdapter, TrialAgentsAdapter
    from generator import generate_patients, summarize_cohort
    from oracle import MelanomaOutcomeOracle, parse_treatment


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "data" / "results"


def _load_patients(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of patients in {path}")
    return data


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def _save_patients(patients: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_patients = _load_json_list(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(existing_patients + patients, f, indent=2, ensure_ascii=False)


def _evaluate_arm(
    oracle: MelanomaOutcomeOracle,
    params: Dict[str, Any],
    adapter_name: str,
    decision: str,
    basis: str,
) -> Dict[str, Any]:
    evaluation = oracle.calculate_event_free_probability(params, decision)
    return {
        "name": adapter_name,
        "decision": decision,
        "basis": basis,
        "endpoint": evaluation["endpoint"],
        "pfs_12": evaluation["pfs_12"],
        "base_probability": evaluation["base_probability"],
        "cumulative_hr": evaluation["cumulative_hr"],
        "parsed_treatment": evaluation["parsed_treatment"],
        "evaluation_log": evaluation["evaluation_log"],
        "evidence": evaluation["evidence"],
    }


def run_virtual_trial(
    n: int = 100,
    seed: int = 42,
    patient_path: str | None = None,
    out_dir: str | None = None,
    enrich_edge_cases: bool = True,
    trialagents_mode: str = "rule_based",
    trialagents_llm_model: str = "deepseek-chat",
    trialagents_doctor_num: int = 4,
    trialagents_max_round: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run SoC, baseline, and TrialAgents-style decisions on one cohort."""

    output_dir = Path(out_dir) if out_dir else DEFAULT_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(patient_path):
        patients = _load_patients(Path(patient_path))
        if n:
            patients = patients[:n]
    else:
        patients = generate_patients(n=n, seed=seed, enrich_edge_cases=enrich_edge_cases)
        _save_patients(patients, patient_path)

    soc = SocAdapter()
    baseline = BaselineSocAdapter()
    trialagents = TrialAgentsAdapter(
        mode=trialagents_mode,
        llm_model=trialagents_llm_model,
        doctor_num=trialagents_doctor_num,
        max_round=trialagents_max_round,
    )
    oracle = MelanomaOutcomeOracle(eval_time_months=12)

    rows: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []

    for patient in patients:
        params = patient["hidden_params"]
        text_profile = patient["text_profile"]
        patient_id = params.get("id", "unknown")

        soc_decision, soc_basis = soc.recommend(params, text_profile)
        base_decision, base_basis = baseline.recommend(params, text_profile)
        agent_decision, agent_basis = trialagents.recommend(params, text_profile)

        soc_eval = _evaluate_arm(oracle, params, "soc", soc_decision, soc_basis)
        base_eval = _evaluate_arm(oracle, params, "baseline", base_decision, base_basis)
        agent_eval = _evaluate_arm(oracle, params, "trialagents", agent_decision, agent_basis)

        row = {
            "patient_id": patient_id,
            "stage": params.get("stage"),
            "subtype": params.get("subtype"),
            "genotype": params.get("genotype"),
            "edge_case_tags": "|".join(params.get("edge_case_tags", [])),
            "soc_decision": soc_decision,
            "base_decision": base_decision,
            "agent_decision": agent_decision,
            "endpoint": soc_eval["endpoint"],
            "soc_pfs_12": round(float(soc_eval["pfs_12"]), 4),
            "base_pfs_12": round(float(base_eval["pfs_12"]), 4),
            "agent_pfs_12": round(float(agent_eval["pfs_12"]), 4),
            "agent_delta_vs_soc": round(float(agent_eval["pfs_12"] - soc_eval["pfs_12"]), 4),
            "agent_delta_vs_base": round(float(agent_eval["pfs_12"] - base_eval["pfs_12"]), 4),
            "agent_better_than_soc": agent_eval["pfs_12"] > soc_eval["pfs_12"] + 0.001,
            "agent_better_than_base": agent_eval["pfs_12"] > base_eval["pfs_12"] + 0.001,
        }
        rows.append(row)
        reports.append(
            {
                "patient_id": patient_id,
                "text_profile": text_profile,
                "hidden_params": params,
                "arms": {
                    "soc": soc_eval,
                    "baseline": base_eval,
                    "trialagents": agent_eval,
                },
            }
        )

    _write_results(rows, reports, patients, output_dir)
    return rows, reports


def _write_results(
    rows: List[Dict[str, Any]],
    reports: List[Dict[str, Any]],
    patients: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "trial_results.csv"
    json_path = output_dir / "reports.json"
    summary_path = output_dir / "summary.json"

    if rows:
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    existing_reports = _load_json_list(json_path)
    all_reports = existing_reports + reports
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    all_rows = _load_csv_rows(csv_path)
    all_patients = [
        {"hidden_params": report["hidden_params"], "text_profile": report["text_profile"]}
        for report in all_reports
        if "hidden_params" in report and "text_profile" in report
    ]
    summary = summarize_results(all_rows or rows, all_patients or patients)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def summarize_results(rows: List[Dict[str, Any]], patients: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "cohort": summarize_cohort(patients)}

    avg = lambda key: sum(float(row[key]) for row in rows) / n
    edge_rows = [row for row in rows if row["edge_case_tags"]]
    edge_n = len(edge_rows)
    return {
        "n": n,
        "cohort": summarize_cohort(patients),
        "mean_soc_pfs_12": round(avg("soc_pfs_12"), 4),
        "mean_base_pfs_12": round(avg("base_pfs_12"), 4),
        "mean_agent_pfs_12": round(avg("agent_pfs_12"), 4),
        "mean_agent_delta_vs_soc": round(avg("agent_delta_vs_soc"), 4),
        "mean_agent_delta_vs_base": round(avg("agent_delta_vs_base"), 4),
        "agent_better_than_soc_count": sum(_as_bool(row["agent_better_than_soc"]) for row in rows),
        "agent_better_than_base_count": sum(_as_bool(row["agent_better_than_base"]) for row in rows),
        "edge_case_n": edge_n,
        "edge_case_agent_better_than_soc_count": sum(_as_bool(row["agent_better_than_soc"]) for row in edge_rows),
    }


def run_sanity_checks() -> Dict[str, Any]:
    """Small deterministic checks for parser and oracle behavior."""

    oracle = MelanomaOutcomeOracle()
    cases = [
        (
            {
                "id": "sanity-braf",
                "stage": "IV",
                "subtype": "cutaneous",
                "braf_mutation": True,
                "kit_mutation": False,
                "ecog": 0,
                "ldh_ratio": 1.0,
                "tumor_burden": "low",
            },
            "Targeted_Dual",
            "Immuno_Single",
            "BRAF targeted should outperform non-specific single IO in BRAF-mutant disease.",
        ),
        (
            {
                "id": "sanity-liver-high-burden",
                "stage": "IV",
                "subtype": "cutaneous",
                "braf_mutation": False,
                "kit_mutation": False,
                "liver_metastases": True,
                "ecog": 0,
                "ldh_ratio": 1.6,
                "tumor_burden": "high",
                "tumor_burden_score": 0.75,
            },
            "Immuno_Dual",
            "Immuno_Single",
            "Dual IO should outperform single IO in fit cutaneous melanoma with liver metastases/high burden.",
        ),
        (
            {
                "id": "sanity-autoimmune",
                "stage": "IV",
                "subtype": "cutaneous",
                "braf_mutation": False,
                "kit_mutation": False,
                "autoimmune_disease": True,
                "ecog": 0,
                "ldh_ratio": 1.0,
                "tumor_burden": "low",
            },
            "Supportive + Chemo",
            "Immuno_Dual",
            "Autoimmune toxicity should penalize dual IO.",
        ),
        (
            {
                "id": "sanity-brain",
                "stage": "IV",
                "subtype": "cutaneous",
                "braf_mutation": True,
                "kit_mutation": False,
                "brain_metastases": True,
                "symptomatic_brain_metastases": True,
                "ecog": 1,
                "ldh_ratio": 1.0,
                "tumor_burden": "high",
            },
            "SRS + Targeted_Dual",
            "Targeted_Dual",
            "Symptomatic CNS disease should benefit from explicit SRS/local therapy.",
        ),
    ]

    results: Dict[str, Any] = {
        "parser": parse_treatment("Surgery + SRS + Targeted_Dual + Supportive").__dict__,
        "cases": [],
    }
    for patient, preferred, comparator, note in cases:
        preferred_eval = oracle.calculate_event_free_probability(patient, preferred)
        comparator_eval = oracle.calculate_event_free_probability(patient, comparator)
        passed = preferred_eval["pfs_12"] > comparator_eval["pfs_12"]
        results["cases"].append(
            {
                "patient_id": patient["id"],
                "preferred": preferred,
                "comparator": comparator,
                "preferred_pfs_12": preferred_eval["pfs_12"],
                "comparator_pfs_12": comparator_eval["pfs_12"],
                "passed": passed,
                "note": note,
            }
        )
    results["passed"] = all(case["passed"] for case in results["cases"]) and results["parser"]["has_srs"] and results["parser"]["has_targeted_dual"]
    return results


def _print_summary(rows: List[Dict[str, Any]], patients: List[Dict[str, Any]]) -> None:
    summary = summarize_results(rows, patients)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rare melanoma virtual trial simulation")
    parser.add_argument("--n", type=int, default=100, help="Number of generated or loaded patients to evaluate.")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic cohort generation.")
    parser.add_argument("--patient-path", type=str, default=None, help="Optional JSON cohort path.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for CSV/JSON results.")
    parser.add_argument("--no-edge-enrichment", default=True, help="Disable edge-case enrichment in the generator.")
    parser.add_argument("--trialagents-mode", type=str, default="single_agent",
                        choices=["rule_based", "single_agent", "multi_agent"],
                        help="TrialAgents recommendation mode: rule_based, single_agent, or multi_agent")
    parser.add_argument("--trialagents-llm-model", type=str, default="deepseek-chat", help="LLM model")
    parser.add_argument("--trialagents-doctor-num", type=int, default=4, help="Number of specialists in multi_agent mode")
    parser.add_argument("--trialagents-max-round", type=int, default=1, help="Max discussion rounds in multi_agent mode")
    parser.add_argument("--sanity", action="store_true", help="Run oracle sanity checks before the virtual trial.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.sanity:
        sanity = run_sanity_checks()
        print(json.dumps(sanity, indent=2, ensure_ascii=False))
        if not sanity["passed"]:
            raise SystemExit("Sanity checks failed")

    # seeds = [
    #     42, 100, 101, 102, 103, 104, 105, 106, 107, 108,
    #     109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    #     120, 121, 122, 123, 124, 1, 3407, 2025, 2026, 2030
    # ]
    seeds = [120, 121, 122, 123, 124]
    # print(f"Running virtual trial with seeds: {seeds}")
    for seed in seeds:
        args.out_dir = ROOT / "data" / f"results_seed_{seed}_num_{args.n}" / f"{args.trialagents_mode}"
        os.makedirs(args.out_dir, exist_ok=True)
        args.patient_path = ROOT / "data" / f"results_seed_{seed}_num_{args.n}" / "patients.json"
        start_time = time.time()
        rows, reports = run_virtual_trial(
            n=args.n,
            seed=seed,
            patient_path=args.patient_path,
            out_dir=args.out_dir,
            enrich_edge_cases=not args.no_edge_enrichment,
            trialagents_mode=args.trialagents_mode,
            trialagents_llm_model=args.trialagents_llm_model,
            trialagents_doctor_num=args.trialagents_doctor_num,
            trialagents_max_round=args.trialagents_max_round,
        )
        end_time = time.time()
        print(f"Time taken for seed {seed}: {end_time - start_time} seconds")
        patients = [{"hidden_params": report["hidden_params"], "text_profile": report["text_profile"]} for report in reports]
        _print_summary(rows, patients)


if __name__ == "__main__":
    main()
