"""
TrialAgents Phase 1: Clinical Review Generation

Three Clinician Agents generate initial clinical reviews in parallel.
"""

import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..agents import ClinicianAgent
from utils.config import DATASET_CONFIGS, AGENT_CONFIG, DEFAULT_LLM
from utils.data.paths import build_experiment_path, ensure_dir

try:
    from utils.adapters import RetrievalSystem, wrap_data
except ImportError:
    RetrievalSystem = None
    wrap_data = None


def run_phase1(
    dataset_name: str = "NCT00694382",
    task: str = "mortality_prediction",
    llm: str = None,
    group: str = "experimental",
    start_idx: int = 0,
    end_idx: int = None,
):
    """
    Execute Phase 1: Clinical Review Generation.

    Args:
        dataset_name: Dataset name
        task: Task name
        llm: LLM model name
        group: Patient group
        start_idx: Start index
        end_idx: End index
    """
    config = {
        "dataset_name": dataset_name,
        "task": task,
        "llm": llm or DEFAULT_LLM,
        "group": group,
        "clinician_num": AGENT_CONFIG.get("clinician_num", 3),
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
    ensure_dir(save_dir)

    print(f"\nPhase 1: Clinical Review Generation")
    print(f"Dataset: {dataset_name}, Task: {task}, LLM: {config['llm']}")
    print(f"Output: {save_dir}\n")

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

    clinician_agents = [
        ClinicianAgent(config, i, retrieval_system, llm=config["llm"], save_dir=save_dir)
        for i in range(config["clinician_num"])
    ]

    for patient_index, patient_df in tqdm(
        enumerate(patients_list),
        total=len(patients_list),
        desc="Phase 1 - Generating reviews"
    ):
        patient_id = patient_df[0]
        sub_save_dir = f"{save_dir}/{patient_id}"

        review_file = os.path.join(sub_save_dir, "clinician1_review.json")
        if os.path.exists(review_file):
            print(f"Skipping {patient_id} - already processed")
            continue

        ensure_dir(sub_save_dir)
        print(f"\nProcessing {patient_id}...")

        def process_clinician(clinician_agent, idx):
            response, context, messages, p_token, c_token = clinician_agent.analysis(
                patient_index, patient_id
            )

            with open(f"{sub_save_dir}/clinician{idx + 1}_review.json", "w") as f:
                json.dump(response, f, indent=2)

            with open(f"{sub_save_dir}/clinician{idx + 1}_review_messages.json", "w") as f:
                json.dump(messages, f, indent=2)

            if len(messages) > 1:
                with open(f"{sub_save_dir}/clinician{idx + 1}_review_userprompt.txt", "w") as f:
                    f.write(messages[1].get("content", ""))

            return response

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_clinician, clinician_agents[i], i)
                for i in range(config["clinician_num"])
            ]
            results = [future.result() for future in futures]

        print(f"Generated {len(results)} clinical reviews for {patient_id}")

    print(f"\nPhase 1 completed. Results saved to {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Clinical Review Generation")
    parser.add_argument("--dataset", default="NCT00694382", help="Dataset name")
    parser.add_argument("--task", default="mortality_prediction", help="Task name")
    parser.add_argument("--llm", default=None, help="LLM model")
    parser.add_argument("--group", default="experimental", help="Patient group")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=None, help="End index")

    args = parser.parse_args()

    run_phase1(
        dataset_name=args.dataset,
        task=args.task,
        llm=args.llm,
        group=args.group,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == "__main__":
    main()
