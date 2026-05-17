#!/usr/bin/env python3
"""
MAS Round Statistics Collector

Parses patient directories under the experimental response folder and generates
round-level classification statistics (all_correct / all_incorrect / disagreement)
for Initial, R1, and Final stages.

Output files (in OUTPUT_DIR):
- patient_round_details.csv : per-patient per-round details
- round_summary.csv         : aggregated counts and percentages
- round_summary.json        : same data in JSON format
- skipped_patients.csv      : patients skipped with reasons
"""

import json
import os
import glob
import re
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

# Configuration
DATA_ROOT = Path("/home/chunyu.lei/DEAP/SELF_DATA/mas/m-agents/response/NCT00694382_MRP_deepseek-chat_MSD_experimental")
OUTPUT_DIR = Path("/home/chunyu.lei/DEAP/SELF_DATA/mas/TrialAgents/utils/outputs/mas_round_stats")

THRESHOLD = 0.5
CLINICIAN_FILES = ["clinician1_review.json", "clinician2_review.json", "clinician3_review.json"]
CLINICIAN_IDS = ["clinician1", "clinician2", "clinician3"]
SUPERVISOR_FILE = "supervisor_logits_json.json"


def logit_to_label(logit: float) -> int:
    """Convert logit to binary label using threshold."""
    return 1 if logit >= THRESHOLD else 0


def parse_clinician_initial(path: Path) -> Dict[str, Any]:
    """Parse a clinician initial review file (clinician*_review.json)."""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "logit": data.get("logit"),
        "conf": data.get("conf_logit"),
    }


def parse_clinician_round(path: Path, prev_logit: float, prev_conf: float) -> Dict[str, Any]:
    """
    Parse a clinician round review file.
    - opinion == "changed": use new_logit / new_conf_logit
    - opinion == "not_changed": inherit from prev state
    - fallback: use logit / conf_logit if present
    """
    with open(path, "r") as f:
        data = json.load(f)

    opinion = data.get("opinion", "")

    if opinion == "changed":
        return {
            "logit": data.get("new_logit"),
            "conf": data.get("new_conf_logit"),
            "used_new": True,
            "opinion": opinion,
            "new_logit": data.get("new_logit"),
            "new_conf_logit": data.get("new_conf_logit"),
        }
    elif opinion == "not_changed":
        return {
            "logit": prev_logit,
            "conf": prev_conf,
            "used_new": False,
            "inherited": True,
            "opinion": opinion,
        }
    else:
        # Fallback: use direct logit/conf_logit if present
        if "logit" in data and "conf_logit" in data:
            return {
                "logit": data.get("logit"),
                "conf": data.get("conf_logit"),
                "used_new": True,
                "opinion": opinion,
            }
        else:
            # Fallback to previous state if no usable values
            warnings.warn(f"Round file {path} has no opinion or logit fields, inheriting from previous state")
            return {
                "logit": prev_logit,
                "conf": prev_conf,
                "used_new": False,
                "inherited": True,
                "opinion": opinion,
            }


def discover_round_files(patient_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    Discover all clinician round review files in one patient directory.
    Returns mapping: {round_idx: {clinician_id: file_path}}
    """
    round_map: Dict[int, Dict[str, Path]] = defaultdict(dict)
    pattern = str(patient_dir / "clinician*_round*_review.json")
    for file_path in glob.glob(pattern):
        name = Path(file_path).name
        match = re.match(r"^(clinician[1-3])_round(\d+)_review\.json$", name)
        if not match:
            continue
        clinician_id, round_idx_str = match.groups()
        round_idx = int(round_idx_str)
        round_map[round_idx][clinician_id] = Path(file_path)
    return {k: round_map[k] for k in sorted(round_map.keys())}


def parse_supervisor_final(path: Path) -> Dict[str, Any]:
    """Parse supervisor_logits_json.json for final predictions and ground truth."""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "final_logit": data.get("final_logit"),
        "final_conf": data.get("final_conf"),
        "ground_truth": data.get("ground_truth"),
        "clinician0_logit": data.get("clinician0_logit"),
        "clinician0_conf": data.get("clinician0_conf"),
        "clinician1_logit": data.get("clinician1_logit"),
        "clinician1_conf": data.get("clinician1_conf"),
        "clinician2_logit": data.get("clinician2_logit"),
        "clinician2_conf": data.get("clinician2_conf"),
        "meta_logit": data.get("meta_logit"),
        "meta_conf": data.get("meta_conf"),
    }


def classify_round(predictions: List[int], ground_truth: int) -> str:
    """
    Classify a round based on all predictions vs ground truth.
    - all_correct: all predictions match ground truth
    - all_incorrect: all predictions don't match
    - disagreement: mix of correct and incorrect
    """
    if not predictions:
        return "unknown"

    correct_count = sum(1 for p in predictions if p == ground_truth)
    incorrect_count = len(predictions) - correct_count

    if correct_count == len(predictions):
        return "all_correct"
    elif incorrect_count == len(predictions):
        return "all_incorrect"
    else:
        return "disagreement"


def build_round_result(
    clinician_states: Dict[str, Dict[str, Any]],
    ground_truth: int,
    include_change_flags: bool = False,
) -> Dict[str, Any]:
    """Build one round's detail payload from doctor states."""
    logits = [clinician_states[d]["logit"] for d in CLINICIAN_IDS]
    confs = [clinician_states[d]["conf"] for d in CLINICIAN_IDS]

    if any(logit is None for logit in logits):
        predictions: List[int] = []
    else:
        
        predictions = [logit_to_label(logit) for logit in logits]

    round_result = {
        "clinician1_logit": logits[0],
        "clinician1_conf": confs[0],
        "clinician2_logit": logits[1],
        "clinician2_conf": confs[1],
        "clinician3_logit": logits[2],
        "clinician3_conf": confs[2],
        "predictions": predictions,
        "classification": classify_round(predictions, ground_truth),
    }

    if include_change_flags:
        for clinician_id in CLINICIAN_IDS:
            round_result[f"{clinician_id}_changed"] = clinician_states[clinician_id].get("used_new", False)
            round_result[f"{clinician_id}_opinion"] = clinician_states[clinician_id].get("opinion")

    return round_result


def process_patient(patient_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Process a single patient directory.
    Returns a dict with round details or None if skipped.
    """
    missing_files = []

    # Check required initial files
    initial_states: Dict[str, Dict[str, Any]] = {}
    for doc_file in CLINICIAN_FILES:
        path = patient_dir / doc_file
        if not path.exists():
            missing_files.append(f"missing {doc_file}")
            continue
        try:
            clinician_id = doc_file.replace("_review.json", "")
            initial_states[clinician_id] = parse_clinician_initial(path)
        except Exception as e:
            missing_files.append(f"error reading {doc_file}: {e}")
            continue

    if len(initial_states) != len(CLINICIAN_IDS):
        return {"skipped": True, "reason": f"Missing initial files: {', '.join(missing_files)}"}

    # Discover and parse all rounds (optional - some patients may not have rounds)
    round_file_map = discover_round_files(patient_dir)

    # Check supervisor file
    supervisor_path = patient_dir / SUPERVISOR_FILE
    if not supervisor_path.exists():
        return {"skipped": True, "reason": f"missing {SUPERVISOR_FILE}"}

    try:
        supervisor_data = parse_supervisor_final(supervisor_path)
    except Exception as e:
        return {"skipped": True, "reason": f"error reading {SUPERVISOR_FILE}: {e}"}

    ground_truth = supervisor_data.get("ground_truth")
    if ground_truth is None:
        return {"skipped": True, "reason": "ground_truth missing in supervisor file"}

    # Build round details
    patient_id = patient_dir.name
    result = {
        "patient_id": patient_id,
        "ground_truth": ground_truth,
        "rounds": {},
        "supervisor_data": supervisor_data,
        "skipped": False,
    }

    # --- Initial round ---
    result["rounds"]["Initial"] = build_round_result(initial_states, ground_truth, include_change_flags=False)

    # --- Dynamic round reconstruction: R1, R2, ... ---
    current_states = {
        clinician_id: {
            "logit": initial_states[clinician_id]["logit"],
            "conf": initial_states[clinician_id]["conf"],
        }
        for clinician_id in CLINICIAN_IDS
    }

    for round_idx, clinician_file_map in round_file_map.items():
        round_name = f"R{round_idx}"
        round_states: Dict[str, Dict[str, Any]] = {}
        for clinician_id in CLINICIAN_IDS:
            prev = current_states[clinician_id]
            round_file = clinician_file_map.get(clinician_id)
            if round_file is None:
                parsed_state = {
                    "logit": prev["logit"],
                    "conf": prev["conf"],
                    "used_new": False,
                    "inherited": True,
                    "opinion": "missing_inherited",
                }
            else:
                try:
                    parsed_state = parse_clinician_round(round_file, prev["logit"], prev["conf"])
                except Exception as e:
                    warnings.warn(f"Error reading {round_file.name}: {e}")
                    parsed_state = {
                        "logit": prev["logit"],
                        "conf": prev["conf"],
                        "used_new": False,
                        "inherited": True,
                        "opinion": "error_inherited",
                    }

            round_states[clinician_id] = parsed_state
            current_states[clinician_id] = {
                "logit": parsed_state["logit"],
                "conf": parsed_state["conf"],
            }

        result["rounds"][round_name] = build_round_result(round_states, ground_truth, include_change_flags=True)

    # --- Final round ---
    final_logit = supervisor_data.get("final_logit")
    final_conf = supervisor_data.get("final_conf")
    final_label = logit_to_label(final_logit) if final_logit is not None else None

    result["rounds"]["Final"] = {
        "final_logit": final_logit,
        "final_conf": final_conf,
        "final_label": final_label,
        "predictions": [final_label] if final_label is not None else [],
        "classification": classify_round([final_label], ground_truth) if final_label is not None else "unknown",
    }

    return result


def collect_all_patients():
    """Collect statistics from all patient directories."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    patient_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories")

    all_details = []
    patient_results = []
    skipped = []
    round_summary = defaultdict(lambda: {"all_correct": 0, "all_incorrect": 0, "disagreement": 0, "unknown": 0})

    for patient_dir in patient_dirs:
        result = process_patient(patient_dir)

        if result.get("skipped"):
            skipped.append({
                "patient_id": patient_dir.name,
                "reason": result["reason"],
            })
            continue
        patient_results.append(result)

        # Extract details for this patient
        for round_name, round_data in result["rounds"].items():
            row = {
                "patient_id": result["patient_id"],
                "round": round_name,
                "ground_truth": result["ground_truth"],
                "clinician1_logit": round_data.get("clinician1_logit"),
                "clinician1_conf": round_data.get("clinician1_conf"),
                "clinician2_logit": round_data.get("clinician2_logit"),
                "clinician2_conf": round_data.get("clinician2_conf"),
                "clinician3_logit": round_data.get("clinician3_logit"),
                "clinician3_conf": round_data.get("clinician3_conf"),
                "final_logit": round_data.get("final_logit"),
                "final_conf": round_data.get("final_conf"),
                "clinician1_pred": round_data.get("predictions", [None])[0] if len(round_data.get("predictions", [])) > 0 else None,
                "clinician2_pred": round_data.get("predictions", [None])[1] if len(round_data.get("predictions", [])) > 1 else None,
                "clinician3_pred": round_data.get("predictions", [None])[2] if len(round_data.get("predictions", [])) > 2 else None,
                "final_pred": round_data.get("final_label"),
                "clinician1_changed": round_data.get("clinician1_changed"),
                "clinician2_changed": round_data.get("clinician2_changed"),
                "clinician3_changed": round_data.get("clinician3_changed"),
                "clinician1_opinion": round_data.get("clinician1_opinion"),
                "clinician2_opinion": round_data.get("clinician2_opinion"),
                "clinician3_opinion": round_data.get("clinician3_opinion"),
                "classification": round_data.get("classification"),
            }
            all_details.append(row)

            # Update summary
            classification = round_data.get("classification", "unknown")
            round_summary[round_name][classification] += 1

    return all_details, round_summary, skipped, patient_results


def round_sort_key(round_name: str) -> Tuple[int, int]:
    """Sort rounds as Initial -> R1..Rn -> Final -> others."""
    if round_name == "Initial":
        return (0, 0)
    if round_name == "Final":
        return (2, 0)
    match = re.match(r"^R(\d+)$", round_name)
    if match:
        return (1, int(match.group(1)))
    return (3, 0)


def run_targeted_sanity_checks(patient_results: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """
    Plan-required sanity checks:
    1) Verify not_changed inheritance on patient 006521-000-901-000 (R1).
    2) Verify there are at least 2 patients with changed opinions.
    """
    report: Dict[str, Any] = {}

    # Check 1: specific patient inheritance
    sample_id = "006521-000-901-000"
    sample_result = next((r for r in patient_results if r["patient_id"] == sample_id), None)
    sample_section: Dict[str, Any] = {"patient_id": sample_id, "found": sample_result is not None}
    if sample_result is not None:
        initial_data = sample_result["rounds"].get("Initial", {})
        r1_data = sample_result["rounds"].get("R1", {})
        doc_checks = []
        check_pass = True
        for clinician_id in CLINICIAN_IDS:
            opinion = r1_data.get(f"{clinician_id}_opinion")
            if opinion != "not_changed":
                continue
            initial_logit = initial_data.get(f"{clinician_id}_logit")
            initial_conf = initial_data.get(f"{clinician_id}_conf")
            r1_logit = r1_data.get(f"{clinician_id}_logit")
            r1_conf = r1_data.get(f"{clinician_id}_conf")
            inherited_ok = (initial_logit == r1_logit) and (initial_conf == r1_conf)
            check_pass = check_pass and inherited_ok
            doc_checks.append(
                {
                    "clinician": clinician_id,
                    "opinion": opinion,
                    "initial_logit": initial_logit,
                    "r1_logit": r1_logit,
                    "initial_conf": initial_conf,
                    "r1_conf": r1_conf,
                    "inheritance_ok": inherited_ok,
                }
            )
        sample_section["not_changed_checks"] = doc_checks
        sample_section["pass"] = check_pass and len(doc_checks) > 0
    report["sample_patient_inheritance_check"] = sample_section

    # Check 2: at least two changed patients
    changed_patient_ids: List[str] = []
    changed_examples: List[Dict[str, Any]] = []
    for patient_result in patient_results:
        patient_id = patient_result["patient_id"]
        patient_has_changed = False
        for round_name, round_data in patient_result["rounds"].items():
            if not round_name.startswith("R"):
                continue
            for clinician_id in CLINICIAN_IDS:
                if round_data.get(f"{clinician_id}_opinion") == "changed":
                    patient_has_changed = True
                    if len(changed_examples) < 6:
                        changed_examples.append(
                            {
                                "patient_id": patient_id,
                                "round": round_name,
                                "clinician": clinician_id,
                                "logit": round_data.get(f"{clinician_id}_logit"),
                                "conf": round_data.get(f"{clinician_id}_conf"),
                            }
                        )
        if patient_has_changed:
            changed_patient_ids.append(patient_id)

    unique_changed_patient_ids = sorted(set(changed_patient_ids))
    report["changed_patient_check"] = {
        "changed_patient_count": len(unique_changed_patient_ids),
        "required_minimum": 2,
        "pass": len(unique_changed_patient_ids) >= 2,
        "sample_patient_ids": unique_changed_patient_ids[:5],
        "examples": changed_examples,
    }

    report_path = output_dir / "sanity_check_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved sanity check report: {report_path}")
    return report


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Collect data
    all_details, round_summary, skipped, patient_results = collect_all_patients()

    # Save patient_round_details.csv
    details_df = pd.DataFrame(all_details)
    details_path = OUTPUT_DIR / "patient_round_details.csv"
    details_df.to_csv(details_path, index=False)
    print(f"Saved patient details: {details_path} ({len(details_df)} rows)")

    # Compute summary statistics
    total_patients = len(details_df["patient_id"].unique()) if not details_df.empty else 0
    summary_data = []

    ordered_rounds = sorted(round_summary.keys(), key=round_sort_key)
    for round_name in ordered_rounds:
        counts = round_summary.get(round_name, {})
        total = sum(counts.values())

        if total > 0:
            all_correct_pct = counts.get("all_correct", 0) / total * 100
            all_incorrect_pct = counts.get("all_incorrect", 0) / total * 100
            disagreement_pct = counts.get("disagreement", 0) / total * 100
            unknown_pct = counts.get("unknown", 0) / total * 100
        else:
            all_correct_pct = all_incorrect_pct = disagreement_pct = unknown_pct = 0.0

        summary_data.append({
            "round": round_name,
            "all_correct": counts.get("all_correct", 0),
            "all_incorrect": counts.get("all_incorrect", 0),
            "disagreement": counts.get("disagreement", 0),
            "unknown": counts.get("unknown", 0),
            "total": total,
            "all_correct_pct": round(all_correct_pct, 2),
            "all_incorrect_pct": round(all_incorrect_pct, 2),
            "disagreement_pct": round(disagreement_pct, 2),
            "unknown_pct": round(unknown_pct, 2),
        })

    # Save round_summary.csv
    summary_df = pd.DataFrame(summary_data)
    summary_path = OUTPUT_DIR / "round_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved round summary: {summary_path}")
    print(summary_df.to_string(index=False))

    # Save round_summary.json
    summary_json = {
        "threshold": THRESHOLD,
        "total_valid_patients": total_patients,
        "rounds": {row["round"]: row for row in summary_data},
    }
    json_path = OUTPUT_DIR / "round_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved JSON summary: {json_path}")

    # Plan-required targeted sanity checks
    sanity_report = run_targeted_sanity_checks(patient_results, OUTPUT_DIR)
    print(f"Sanity check (sample inheritance): {sanity_report['sample_patient_inheritance_check'].get('pass')}")
    print(f"Sanity check (>=2 changed patients): {sanity_report['changed_patient_check'].get('pass')}")

    # Save skipped_patients.csv
    skipped_df = pd.DataFrame(skipped)
    if not skipped_df.empty:
        skipped_path = OUTPUT_DIR / "skipped_patients.csv"
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Saved skipped patients: {skipped_path} ({len(skipped_df)} patients)")

    # Verification: check sums
    print(f"\n=== Verification ===")
    print(f"Total valid patients: {total_patients}")
    for round_name in ordered_rounds:
        total = sum(round_summary.get(round_name, {}).values())
        print(f"  {round_name}: {total} entries (should equal {total_patients})")

    return all_details, round_summary, skipped


if __name__ == "__main__":
    main()
