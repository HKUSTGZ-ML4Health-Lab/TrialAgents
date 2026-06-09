"""
TrialAgents Phase 3: Diverse Task Completion

Executes specific tasks based on the final report
(e.g., mortality prediction / digital twin generation).
"""

import json
import os
import argparse
from tqdm import tqdm

from ..agents import ClinicianAgent, SupervisorAgent, SimulatorAgent
from utils.config import DATASET_CONFIGS, AGENT_CONFIG, DEFAULT_LLM
from utils.data.paths import build_experiment_path, ensure_dir

try:
    from utils.adapters import get_final_result
except ImportError:
    get_final_result = None


def run_phase3(
    dataset_name: str = "NCT00694382",
    task: str = "mortality_prediction",
    llm: str = None,
    group: str = "experimental",
    start_idx: int = 0,
    end_idx: int = None,
):
    """Execute Phase 3: Diverse Task Completion."""
    config = {
        "dataset_name": dataset_name,
        "task": task,
        "llm": llm or DEFAULT_LLM,
        "group": group,
        "clinician_num": AGENT_CONFIG.get("clinician_num", 3),
        "retriever_name": "MedCPT",
        "corpus_name": "MSD",
        "mode": task,
    }

    save_dir = build_experiment_path(
        dataset_name=config["dataset_name"],
        task=config["task"],
        llm=config["llm"],
        corpus_name=config["corpus_name"],
        group=config["group"],
    )

    print(f"\nPhase 3: Diverse Task Completion - {task}")
    print(f"Dataset: {dataset_name}\n")

    dataset_info = DATASET_CONFIGS.get(dataset_name, {})
    total = dataset_info.get("num_patients", 100)

    if end_idx is None:
        end_idx = total

    supervisor_agent = SupervisorAgent(llm=config["llm"])
    clinician_agents = [
        ClinicianAgent(config, i, None, llm=config["llm"], save_dir=save_dir)
        for i in range(config["clinician_num"])
    ]

    results = {}

    for idx in tqdm(range(start_idx, end_idx), desc="Phase 3"):
        patient_id = f"{dataset_name}_{idx:03d}"
        sub_save_dir = f"{save_dir}/{patient_id}"

        if not os.path.exists(os.path.join(sub_save_dir, "meta_final_summary.json")):
            print(f"Skipping {patient_id} - Phase 2 not completed")
            continue

        if os.path.exists(os.path.join(sub_save_dir, "supervisor_logits_json.json")):
            print(f"Skipping {patient_id} - already processed")
            continue

        print(f"\nProcessing {patient_id}...")

        latest_analysis = []
        for i, clinician_agent in enumerate(clinician_agents):
            ans = clinician_agent.get_latest_analysis(sub_save_dir, i + 1)
            latest_analysis.append(ans)

        logits, messages, _, _ = supervisor_agent.revise_logits(latest_analysis)

        with open(f"{sub_save_dir}/supervisor_userprompt.txt", "w") as f:
            if len(messages) > 1:
                f.write(messages[1].get("content", ""))

        if get_final_result is not None:
            logits['final_logit'], logits['final_conf'] = get_final_result(logits)
        else:
            logits['final_logit'] = logits.get('logit', 0.5)
            logits['final_conf'] = logits.get('conf_logit', 0.5)

        logits['ground_truth'] = 0  # needs to be loaded from data

        with open(f"{sub_save_dir}/supervisor_logits_json.json", "w") as f:
            json.dump(logits, f, indent=2)

        if task == "digital_twin":
            final_report = {}
            simulator = SimulatorAgent(llm=config["llm"])
            digital_twin = simulator.simulate(final_report)

            with open(f"{sub_save_dir}/digital_twin.json", "w") as f:
                json.dump(digital_twin, f, indent=2)

        results[patient_id] = {
            "final_prediction": logits.get('final_logit'),
            "final_confidence": logits.get('final_conf'),
        }

        print(f"Final prediction: {logits.get('final_logit')}")

    with open(f"{save_dir}/phase3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPhase 3 completed. Results in {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Diverse Task Completion")
    parser.add_argument("--dataset", default="NCT00694382")
    parser.add_argument("--task", default="mortality_prediction",
                       choices=["mortality_prediction", "digital_twin"])
    parser.add_argument("--llm", default=None)
    parser.add_argument("--group", default="experimental")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    args = parser.parse_args()
    run_phase3(args.dataset, args.task, args.llm, args.group, args.start_idx, args.end_idx)


if __name__ == "__main__":
    main()
