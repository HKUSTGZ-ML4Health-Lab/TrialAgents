"""
TrialAgents Phase 2: Multi-round Collaborative Diagnosis

Supervisor Agent organizes Clinician Agents for multi-round discussion.
"""

import json
import os
import argparse
from tqdm import tqdm

from ..agents import ClinicianAgent, SupervisorAgent
from ..collaboration import CollaborationEngine
from utils.config import DATASET_CONFIGS, AGENT_CONFIG, DEFAULT_LLM
from utils.data.paths import build_experiment_path, ensure_dir

try:
    from utils.adapters import RetrievalSystem, wrap_data
except ImportError:
    RetrievalSystem = None
    wrap_data = None


def run_phase2(
    dataset_name: str = "NCT00694382",
    task: str = "mortality_prediction",
    llm: str = None,
    group: str = "experimental",
    start_idx: int = 0,
    end_idx: int = None,
):
    """Execute Phase 2: Multi-round Collaborative Diagnosis."""
    config = {
        "dataset_name": dataset_name,
        "task": task,
        "llm": llm or DEFAULT_LLM,
        "group": group,
        "clinician_num": AGENT_CONFIG.get("clinician_num", 3),
        "max_round": AGENT_CONFIG.get("max_round", 3),
        "retriever_name": "MedCPT",
        "corpus_name": "MSD",
        "mode": task,
        "model_names": ["time_series_model"] * AGENT_CONFIG.get("clinician_num", 3),
    }

    save_dir = build_experiment_path(
        dataset_name=config["dataset_name"],
        task=config["task"],
        llm=config["llm"],
        corpus_name=config["corpus_name"],
        group=config["group"],
    )

    print(f"\nPhase 2: Multi-round Collaborative Diagnosis")
    print(f"Dataset: {dataset_name}, Max Rounds: {config['max_round']}\n")

    if wrap_data is None:
        raise ImportError("wrap_data not available")

    patients_list = wrap_data(config)
    dataset_info = DATASET_CONFIGS.get(dataset_name, {})
    total = dataset_info.get("num_patients", len(patients_list))

    if end_idx is None:
        end_idx = total

    patients_list = patients_list[start_idx:end_idx]

    if RetrievalSystem is not None:
        retrieval_system = RetrievalSystem(
            retriever_name=config["retriever_name"],
            corpus_name=config["corpus_name"]
        )
    else:
        retrieval_system = None

    supervisor_agent = SupervisorAgent(llm=config["llm"])
    clinician_agents = [
        ClinicianAgent(config, i, retrieval_system, llm=config["llm"], save_dir=save_dir)
        for i in range(config["clinician_num"])
    ]

    for patient_index, patient_df in tqdm(
        enumerate(patients_list),
        total=len(patients_list),
        desc="Phase 2 - Collaboration"
    ):
        patient_id = patient_df[0]
        sub_save_dir = f"{save_dir}/{patient_id}"

        review_file = os.path.join(sub_save_dir, "clinician1_review.json")
        if not os.path.exists(review_file):
            print(f"Skipping {patient_id} - Phase 1 not completed")
            continue

        final_summary_file = os.path.join(sub_save_dir, "meta_final_summary.json")
        if os.path.exists(final_summary_file):
            print(f"Skipping {patient_id} - Phase 2 already completed")
            continue

        print(f"\nProcessing {patient_id}...")

        analysis_results = []
        for i in range(config["clinician_num"]):
            with open(f"{sub_save_dir}/clinician{i+1}_review.json", "r") as f:
                analysis_results.append(json.load(f))

        supervisor_agent.set_basic_info("")
        summary_content, messages, _, _ = supervisor_agent.summary(analysis_results, is_initial=True)

        with open(f"{sub_save_dir}/meta_summary.json", "w") as f:
            json.dump(summary_content, f, indent=2)

        collaboration = CollaborationEngine(
            supervisor_agent=supervisor_agent,
            clinician_agents=clinician_agents,
            supervisor_opinion=summary_content.get("answer", ""),
            supervisor_report=summary_content.get("report", ""),
            save_dir=sub_save_dir,
            clinician_num=config["clinician_num"],
            max_round=config["max_round"]
        )

        current_report, rounds, _, _ = collaboration.collaborate()

        with open(f"{sub_save_dir}/meta_final_summary.json", "w") as f:
            json.dump({"final_report": current_report, "rounds": rounds}, f, indent=2)

        print(f"Completed {rounds} rounds for {patient_id}")

    print(f"\nPhase 2 completed. Results in {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Multi-round Collaborative Diagnosis")
    parser.add_argument("--dataset", default="NCT00694382")
    parser.add_argument("--task", default="mortality_prediction")
    parser.add_argument("--llm", default=None)
    parser.add_argument("--group", default="experimental")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    args = parser.parse_args()
    run_phase2(args.dataset, args.task, args.llm, args.group, args.start_idx, args.end_idx)


if __name__ == "__main__":
    main()
