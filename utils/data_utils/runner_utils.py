"""
TrialAgents Data Runner Utilities

Migrated from m-agents/utils/runner_utils.py
Provides time series data processing functionality
"""

import os
import json
from pathlib import Path
from typing import List


def process_time_series(dataset: str, patient_list: List) -> None:
    """
    Serialize patient data and save as JSON

    Args:
        dataset: Dataset name
        patient_list: Patient data list

    Note:
        Data is saved to {MAS_PROCESSED_DATA_DIR}/{dataset}/{patient_id}.json
    """
    _default_dir = str(Path(__file__).parent.parent.parent.parent / "m-agents" / "processed_data")
    save_base = os.getenv("MAS_PROCESSED_DATA_DIR", _default_dir)
    save_base = os.path.join(save_base, dataset)

    if not os.path.exists(save_base):
        os.makedirs(save_base)

    for patient_data in patient_list:
        patient_id = patient_data[0]
        save_path = os.path.join(save_base, f"{patient_id}.json")

        if os.path.exists(save_path):
            continue

        # patient_data format: [patient_id, features..., label]
        # Serialize and save
        try:
            with open(save_path, 'w') as f:
                json.dump(patient_data, f)
        except Exception as e:
            print(f"Error saving patient {patient_id}: {e}")


__all__ = ['process_time_series']
